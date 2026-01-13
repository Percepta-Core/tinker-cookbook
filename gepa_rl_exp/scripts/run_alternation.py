"""
GEPA+RL Alternation: Block coordinate ascent on (s, θ).

Implements the simple alternation loop:
  1. GEPA phase: Evaluate strategies on current θ, pick winner s*
  2. RL phase: Train with s* for N batches, update θ
  3. Repeat

This is the "one-shot GEPA → RL → GEPA → repeat" pattern for joint
optimization over prompt strategy `s` and model weights `θ`.

Usage:
    cd gepa_rl_exp
    python -m scripts.run_alternation env=arithmetic n_rounds=3 rl_batches_per_round=10
    python -m scripts.run_alternation env=gsm8k n_rounds=2 rl_batches_per_round=50
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
import tinker
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.metric_util import dataset_to_env_group_builders
from tinker_cookbook.rl.problem_env import PromptStrategy
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import AsyncConfig, Config, main as rl_main

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompt_strategy import get_strategy, list_strategies
from src.prompt_templates import get_strategy_by_name, get_template
from src.train import get_dataset_builder


@chz.chz
class AlternationConfig:
    """Configuration for GEPA+RL alternation loop on (s, θ)."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None

    # Environment
    env: Literal["arithmetic", "gsm8k", "math"] = "arithmetic"
    seed: int = 0

    # Alternation schedule
    n_rounds: int = 3  # Number of GEPA→RL cycles
    rl_batches_per_round: int = 10  # RL training batches per round (U steps)

    # GEPA settings - strategy pool for `s` optimization
    gepa_num_problems: int = 32  # Problems to evaluate per strategy
    gepa_group_size: int = 4  # Rollouts per problem for GEPA
    strategies: str = "baseline,step_by_step,concise"  # Comma-separated strategy pool
    # Legacy alias for backward compatibility
    templates: str | None = None  # Deprecated: use strategies

    # RL settings (passed through to RL training)
    rl_group_size: int = 4
    rl_groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 256
    temperature: float = 1.0

    # Output
    log_path: str | None = None
    wandb_project: str | None = None

    # Service
    base_url: str | None = None


@dataclass
class CheckpointPaths:
    """Paths to checkpoint artifacts."""

    state_path: str | None  # Full training state (for RL continuation)
    sampler_path: str | None  # Weights only (for GEPA sampling)


@dataclass
class RoundResult:
    """Results from one GEPA→RL round."""

    round_idx: int
    gepa_results: dict[str, float]  # strategy_name -> mean_reward
    selected_strategy: str  # Name of the winning strategy `s*`
    rl_final_correct_rate: float | None
    checkpoint_paths: CheckpointPaths | None

    # Legacy alias for backward compatibility
    @property
    def selected_template(self) -> str:
        return self.selected_strategy


async def run_gepa_phase(
    config: AlternationConfig,
    sampler_path: str | None,
    renderer_name: str,
    strategy_names: list[str],
) -> tuple[PromptStrategy, dict[str, float]]:
    """
    Run GEPA evaluation: evaluate all strategies on current θ.

    This is the `s` optimization step in (s, θ) block coordinate ascent.

    Args:
        sampler_path: Path to sampler weights (for inference only)
        strategy_names: List of strategy names to evaluate

    Returns:
        (winning_strategy, results_dict) where results_dict maps strategy_name -> mean_reward
    """
    print("\n" + "=" * 60)
    print("GEPA PHASE: Evaluating strategies (optimizing s)")
    print("=" * 60)
    print(f"  Current θ: {sampler_path or 'base model (θ_0)'}")
    print(f"  Strategies: {strategy_names}")
    print(f"  Problems per strategy: {config.gepa_num_problems}")

    # Create service client
    service = tinker.ServiceClient()

    # Create sampling client with current weights
    if sampler_path:
        sampling_client = service.create_sampling_client(
            model_path=sampler_path,
            base_model=config.model_name,
        )
    else:
        sampling_client = service.create_sampling_client(base_model=config.model_name)

    results: dict[str, float] = {}
    strategies: dict[str, PromptStrategy] = {}

    for strategy_name in strategy_names:
        print(f"\n  Evaluating strategy: {strategy_name}")

        # Get the strategy object
        strategy = get_strategy_by_name(strategy_name)
        strategies[strategy_name] = strategy

        # Create dataset builder with this strategy
        dataset_builder = get_dataset_builder(
            env=config.env,
            batch_size=1,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.gepa_group_size,
            seed=config.seed,
            prompt_template=strategy_name,  # For backward compat
            n_batches=config.gepa_num_problems,
            strategy=strategy,
        )

        # Build dataset
        train_dataset, _ = await dataset_builder()

        # Get env builders
        all_env_builders = dataset_to_env_group_builders(train_dataset)
        env_builders = all_env_builders[: config.gepa_num_problems]

        # Create policy
        policy = TinkerTokenCompleter(
            sampling_client=sampling_client,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        # Run rollouts
        trajectory_groups = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in env_builders]
        )

        # Compute mean reward
        all_rewards: list[float] = []
        for traj_group in trajectory_groups:
            for traj in traj_group.trajectories_G:
                episode_reward = sum(t.reward for t in traj.transitions)
                all_rewards.append(episode_reward)

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        results[strategy_name] = mean_reward
        print(f"    Mean reward: {mean_reward:.3f}")

    # Select winner (s*)
    winner_name = max(results, key=lambda s: results[s])
    winner = strategies[winner_name]
    print(f"\n  WINNER: s* = {winner_name} (reward={results[winner_name]:.3f})")

    return winner, results


async def run_rl_phase(
    config: AlternationConfig,
    strategy: PromptStrategy,
    round_idx: int,
    state_path: str | None,
    renderer_name: str,
    round_log_path: str,
) -> CheckpointPaths | None:
    """
    Run RL training phase with selected strategy.

    This is the `θ` optimization step in (s, θ) block coordinate ascent.

    Args:
        strategy: The selected prompt strategy s* from GEPA phase
        state_path: Path to full training state (weights + optimizer) for continuation

    Returns:
        CheckpointPaths with both state_path and sampler_path, or None if training failed
    """
    print("\n" + "=" * 60)
    print(f"RL PHASE: Training θ with strategy s* = '{strategy.name}'")
    print("=" * 60)
    print(f"  Batches: {config.rl_batches_per_round}")
    print(f"  Starting θ from: {state_path or 'base model'}")
    print(f"  Log path: {round_log_path}")

    # Build RL config with the strategy
    rl_config = Config(
        learning_rate=config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=config.env,
            batch_size=config.rl_groups_per_batch,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.rl_group_size,
            seed=config.seed + round_idx,  # Vary seed per round
            prompt_template=strategy.name,  # For logging/backward compat
            n_batches=config.rl_batches_per_round,
            strategy=strategy,  # First-class strategy object
        ),
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=f"round{round_idx}-{strategy.name}" if config.wandb_project else None,
        log_path=round_log_path,
        base_url=config.base_url,
        load_checkpoint_path=state_path,
        eval_every=max(1, config.rl_batches_per_round // 2),
        save_every=config.rl_batches_per_round,  # Save at end
    )

    # Run training
    await rl_main(rl_config)

    # Get checkpoint paths from the saved checkpoints
    last_checkpoint = checkpoint_utils.get_last_checkpoint(
        round_log_path, required_key="sampler_path"
    )
    if last_checkpoint:
        return CheckpointPaths(
            state_path=last_checkpoint.get("state_path"),
            sampler_path=last_checkpoint.get("sampler_path"),
        )
    return None


async def run_alternation(config: AlternationConfig) -> list[RoundResult]:
    """Run the full GEPA+RL alternation loop on (s, θ)."""

    print("=" * 70)
    print("GEPA+RL ALTERNATION: Block coordinate ascent on (s, θ)")
    print("=" * 70)
    print(f"Model (θ base): {config.model_name}")
    print(f"Environment: {config.env}")
    print(f"Rounds: {config.n_rounds}")
    print(f"RL batches per round (θ steps): {config.rl_batches_per_round}")
    print("=" * 70)

    # Setup
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Parse strategy pool - support both 'strategies' and legacy 'templates'
    strategy_str = config.templates if config.templates else config.strategies
    strategy_names = [s.strip() for s in strategy_str.split(",")]
    print(f"Strategy pool: {strategy_names}")

    # Determine base log path
    if config.log_path:
        base_log_path = config.log_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wrapper_root = Path(__file__).parent.parent
        base_log_path = str(wrapper_root / "runs" / f"alternation-{timestamp}")

    base_log_path = os.path.expanduser(base_log_path)
    os.makedirs(base_log_path, exist_ok=True)

    # State: track both θ paths and current s
    current_paths: CheckpointPaths | None = None
    results: list[RoundResult] = []

    for round_idx in range(config.n_rounds):
        print(f"\n{'#' * 70}")
        print(f"# ROUND {round_idx + 1} / {config.n_rounds}")
        print(f"{'#' * 70}")

        # 1. GEPA phase: optimize s with θ fixed
        # Use sampler_path for inference
        sampler_path = current_paths.sampler_path if current_paths else None
        selected_strategy, gepa_results = await run_gepa_phase(
            config=config,
            sampler_path=sampler_path,
            renderer_name=renderer_name,
            strategy_names=strategy_names,
        )

        # 2. RL phase: optimize θ with s fixed
        # Use state_path for training continuation
        state_path = current_paths.state_path if current_paths else None
        round_log_path = os.path.join(base_log_path, f"round{round_idx}")
        new_paths = await run_rl_phase(
            config=config,
            strategy=selected_strategy,
            round_idx=round_idx,
            state_path=state_path,
            renderer_name=renderer_name,
            round_log_path=round_log_path,
        )

        # Record results
        result = RoundResult(
            round_idx=round_idx,
            gepa_results=gepa_results,
            selected_strategy=selected_strategy.name,
            rl_final_correct_rate=None,  # Could parse from metrics.jsonl
            checkpoint_paths=new_paths,
        )
        results.append(result)

        # Update state for next round
        current_paths = new_paths

        # Save round summary with both s and θ state
        summary_path = os.path.join(base_log_path, "alternation_log.jsonl")
        with open(summary_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "round": round_idx,
                        "gepa_results": gepa_results,
                        "selected_strategy": selected_strategy.name,
                        # Legacy field for backward compatibility
                        "selected_template": selected_strategy.name,
                        "state_path": new_paths.state_path if new_paths else None,
                        "sampler_path": new_paths.sampler_path if new_paths else None,
                    }
                )
                + "\n"
            )

    return results


def print_summary(results: list[RoundResult]) -> None:
    """Print final summary of alternation run."""
    print("\n" + "=" * 70)
    print("ALTERNATION SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\nRound {r.round_idx}:")
        print(f"  Selected strategy (s*): {r.selected_strategy}")
        print(f"  GEPA scores: {r.gepa_results}")
        if r.checkpoint_paths:
            print(f"  θ state path: {r.checkpoint_paths.state_path}")
            print(f"  θ sampler path: {r.checkpoint_paths.sampler_path}")

    # Track strategy switches
    strategies_used = [r.selected_strategy for r in results]
    switches = sum(1 for i in range(1, len(strategies_used)) if strategies_used[i] != strategies_used[i - 1])
    print(f"\nStrategy switches: {switches}")
    print(f"Strategy trajectory: {' → '.join(strategies_used)}")
    print("=" * 70)


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(AlternationConfig)
    results = asyncio.run(run_alternation(config))
    print_summary(results)


if __name__ == "__main__":
    main()
