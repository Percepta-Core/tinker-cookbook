"""
GEPA+RL Alternation: Block coordinate ascent on (prompt, weights).

Implements the simple alternation loop:
  1. GEPA phase: Evaluate templates on current θ, pick winner
  2. RL phase: Train with winner template for N batches, save checkpoint
  3. Repeat

This is the "one-shot GEPA → RL → GEPA → repeat" pattern described in the design.

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
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.train import AsyncConfig, Config, main as rl_main

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompt_templates import get_template
from src.train import get_dataset_builder


@chz.chz
class AlternationConfig:
    """Configuration for GEPA+RL alternation loop."""

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

    # GEPA settings
    gepa_num_problems: int = 32  # Problems to evaluate per template
    gepa_group_size: int = 4  # Rollouts per problem for GEPA
    templates: str = "baseline,step_by_step,concise"  # Comma-separated template pool

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
    gepa_results: dict[str, float]  # template -> mean_reward
    selected_template: str
    rl_final_correct_rate: float | None
    checkpoint_paths: CheckpointPaths | None


async def run_gepa_phase(
    config: AlternationConfig,
    sampler_path: str | None,
    renderer_name: str,
    templates: list[str],
) -> tuple[str, dict[str, float]]:
    """
    Run GEPA evaluation: evaluate all templates on current checkpoint.

    Args:
        sampler_path: Path to sampler weights (for inference only)

    Returns:
        (winning_template, results_dict)
    """
    print("\n" + "=" * 60)
    print("GEPA PHASE: Evaluating templates")
    print("=" * 60)
    print(f"  Checkpoint: {sampler_path or 'base model (θ_0)'}")
    print(f"  Templates: {templates}")
    print(f"  Problems per template: {config.gepa_num_problems}")

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

    for template in templates:
        print(f"\n  Evaluating: {template}")

        # Create dataset builder for this template
        dataset_builder = get_dataset_builder(
            env=config.env,
            batch_size=1,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.gepa_group_size,
            seed=config.seed,
            prompt_template=template,
            n_batches=config.gepa_num_problems,
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
        results[template] = mean_reward
        print(f"    Mean reward: {mean_reward:.3f}")

    # Select winner
    winner = max(results, key=lambda t: results[t])
    print(f"\n  WINNER: {winner} (reward={results[winner]:.3f})")

    return winner, results


async def run_rl_phase(
    config: AlternationConfig,
    template: str,
    round_idx: int,
    state_path: str | None,
    renderer_name: str,
    round_log_path: str,
) -> CheckpointPaths | None:
    """
    Run RL training phase with selected template.

    Args:
        state_path: Path to full training state (weights + optimizer) for continuation

    Returns:
        CheckpointPaths with both state_path and sampler_path, or None if training failed
    """
    print("\n" + "=" * 60)
    print(f"RL PHASE: Training with template '{template}'")
    print("=" * 60)
    print(f"  Batches: {config.rl_batches_per_round}")
    print(f"  Starting from: {state_path or 'base model'}")
    print(f"  Log path: {round_log_path}")

    # Build RL config
    rl_config = Config(
        learning_rate=config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=config.env,
            batch_size=config.rl_groups_per_batch,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.rl_group_size,
            seed=config.seed + round_idx,  # Vary seed per round
            prompt_template=template,
            n_batches=config.rl_batches_per_round,
        ),
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=f"round{round_idx}-{template}" if config.wandb_project else None,
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
    """Run the full GEPA+RL alternation loop."""

    print("=" * 70)
    print("GEPA+RL ALTERNATION")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Environment: {config.env}")
    print(f"Rounds: {config.n_rounds}")
    print(f"RL batches per round: {config.rl_batches_per_round}")
    print("=" * 70)

    # Setup
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    templates = [t.strip() for t in config.templates.split(",")]

    # Determine base log path
    if config.log_path:
        base_log_path = config.log_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wrapper_root = Path(__file__).parent.parent
        base_log_path = str(wrapper_root / "runs" / f"alternation-{timestamp}")

    base_log_path = os.path.expanduser(base_log_path)
    os.makedirs(base_log_path, exist_ok=True)

    # State: track both state_path (for RL) and sampler_path (for GEPA)
    current_paths: CheckpointPaths | None = None
    results: list[RoundResult] = []

    for round_idx in range(config.n_rounds):
        print(f"\n{'#' * 70}")
        print(f"# ROUND {round_idx + 1} / {config.n_rounds}")
        print(f"{'#' * 70}")

        # 1. GEPA phase: evaluate templates, pick winner
        # Use sampler_path for inference
        sampler_path = current_paths.sampler_path if current_paths else None
        selected_template, gepa_results = await run_gepa_phase(
            config=config,
            sampler_path=sampler_path,
            renderer_name=renderer_name,
            templates=templates,
        )

        # 2. RL phase: train with selected template
        # Use state_path for training continuation
        state_path = current_paths.state_path if current_paths else None
        round_log_path = os.path.join(base_log_path, f"round{round_idx}")
        new_paths = await run_rl_phase(
            config=config,
            template=selected_template,
            round_idx=round_idx,
            state_path=state_path,
            renderer_name=renderer_name,
            round_log_path=round_log_path,
        )

        # Record results
        result = RoundResult(
            round_idx=round_idx,
            gepa_results=gepa_results,
            selected_template=selected_template,
            rl_final_correct_rate=None,  # Could parse from metrics.jsonl
            checkpoint_paths=new_paths,
        )
        results.append(result)

        # Update state for next round
        current_paths = new_paths

        # Save round summary
        summary_path = os.path.join(base_log_path, "alternation_log.jsonl")
        with open(summary_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "round": round_idx,
                        "gepa_results": gepa_results,
                        "selected_template": selected_template,
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
        print(f"  Selected template: {r.selected_template}")
        print(f"  GEPA scores: {r.gepa_results}")
        if r.checkpoint_paths:
            print(f"  State path: {r.checkpoint_paths.state_path}")
            print(f"  Sampler path: {r.checkpoint_paths.sampler_path}")

    # Track template switches
    templates_used = [r.selected_template for r in results]
    switches = sum(1 for i in range(1, len(templates_used)) if templates_used[i] != templates_used[i - 1])
    print(f"\nTemplate switches: {switches}")
    print(f"Templates used: {' → '.join(templates_used)}")
    print("=" * 70)


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(AlternationConfig)
    results = asyncio.run(run_alternation(config))
    print_summary(results)


if __name__ == "__main__":
    main()
