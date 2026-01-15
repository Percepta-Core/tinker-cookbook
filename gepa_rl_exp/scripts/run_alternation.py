"""
GEPA+RL Alternation: Block coordinate ascent on (s, θ).

Implements joint optimization over prompt `s` and model weights `θ`:
  1. GEPA phase: Use trace-based reflection to optimize prompt s
  2. RL phase: Train with optimized s for N batches, update θ
  3. Repeat

This uses true GEPA (arxiv:2507.19457) with LLM reflection on execution
traces, not simple template selection.

Usage:
    cd gepa_rl_exp
    python -m scripts.run_alternation env=arithmetic n_rounds=3 rl_batches_per_round=10
    python -m scripts.run_alternation env=gsm8k n_rounds=2 gepa_iterations=5
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

try:
    import wandb
except ImportError:
    wandb = None
from tinker_cookbook import checkpoint_utils, model_info
from tinker_cookbook.rl.train import Config, main as rl_main

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gepa import GEPAConfig, GEPAOptimizer, GEPACandidate
from src.gepa.config import GEPACandidateMetadata
from src.gepa.reflection import ReflectionClient, DummyReflectionClient
from src.prompt_strategy import SystemPromptStrategy, PromptStrategy
from src.train import get_dataset_builder

# Default seed prompt when GEPA is skipped
DEFAULT_SEED_PROMPT = "Solve the problem step by step. Show your reasoning clearly."


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
    rl_batches_per_round: int = 10  # RL training batches per round
    skip_first_gepa: bool = False  # Skip GEPA in first round (start with RL-only)

    # GEPA settings (trace-based reflection)
    gepa_iterations: int = 3  # GEPA optimization steps per round
    gepa_traces_per_iteration: int = 16  # Problems per GEPA iteration
    gepa_rollouts_per_problem: int = 4  # Rollouts for trace collection

    # Reflection model for GEPA
    reflection_backend: Literal["openai", "anthropic"] = "anthropic"
    reflection_model: str = "claude-opus-4-5-20251101"
    reflection_temperature: float = 0.7

    # RL settings
    rl_group_size: int = 4
    rl_groups_per_batch: int = 100
    learning_rate: float = 5e-4  # Recommended for LoRA (was 1e-5, 50x too low)
    max_tokens: int = 256
    temperature: float = 1.0

    # Output
    log_path: str | None = None
    wandb_project: str | None = "gepa-rl"  # Default wandb project

    # Service
    base_url: str | None = None

    # Debug mode (uses dummy reflection client)
    debug: bool = False


@dataclass
class CheckpointPaths:
    """Paths to checkpoint artifacts."""

    state_path: str | None  # Full training state
    sampler_path: str | None  # Weights only


@dataclass
class RoundResult:
    """Results from one GEPA→RL round."""

    round_idx: int
    gepa_best_prompt: str
    gepa_best_score: float
    gepa_iterations_run: int
    rl_final_correct_rate: float | None
    checkpoint_paths: CheckpointPaths | None


def create_dataset_builder_factory(
    config: AlternationConfig,
    renderer_name: str,
):
    """Create factory for building datasets with a given prompt."""

    def factory(prompt: str):
        strategy = SystemPromptStrategy(
            system_prompt=prompt,
            strategy_name="gepa_candidate",
        )

        return get_dataset_builder(
            env=config.env,
            batch_size=1,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.gepa_rollouts_per_problem,
            seed=config.seed,
            prompt_template="none",
            n_batches=config.gepa_traces_per_iteration,
            strategy=strategy,
        )

    return factory


async def run_gepa_phase(
    config: AlternationConfig,
    sampler_path: str | None,
    renderer_name: str,
    log_path: str,
    seed_prompt: str | None = None,
) -> tuple[PromptStrategy, GEPACandidate]:
    """
    Run GEPA optimization phase: optimize s using trace-based reflection.

    Args:
        sampler_path: Path to sampler weights for current θ
        seed_prompt: Starting prompt (uses previous best or default)

    Returns:
        (optimized_strategy, best_candidate)
    """
    print("\n" + "=" * 60)
    print("GEPA PHASE: Trace-Based Prompt Optimization")
    print("=" * 60)
    print(f"  Current θ: {sampler_path or 'base model'}")
    print(f"  GEPA iterations: {config.gepa_iterations}")
    print(f"  Reflection model: {config.reflection_backend}/{config.reflection_model}")

    # Create reflection client
    if config.debug:
        print("  [DEBUG MODE] Using dummy reflection client")
        reflection_client = DummyReflectionClient()
    else:
        reflection_client = ReflectionClient(
            backend=config.reflection_backend,
            model=config.reflection_model,
            temperature=config.reflection_temperature,
        )

    # Create service client and sampling client
    service = tinker.ServiceClient()

    if sampler_path:
        sampling_client = service.create_sampling_client(
            model_path=sampler_path,
            base_model=config.model_name,
        )
    else:
        sampling_client = service.create_sampling_client(base_model=config.model_name)

    # Create dataset builder factory
    dataset_builder_factory = create_dataset_builder_factory(config, renderer_name)

    # Create GEPA config
    gepa_config = GEPAConfig(
        reflection_backend=config.reflection_backend,
        reflection_model=config.reflection_model,
        reflection_temperature=config.reflection_temperature,
        max_iterations=config.gepa_iterations,
        traces_per_iteration=config.gepa_traces_per_iteration,
        rollouts_per_problem=config.gepa_rollouts_per_problem,
        seed_prompt=seed_prompt,
        seed=config.seed,
    )

    # Create and run optimizer
    optimizer = GEPAOptimizer(
        config=gepa_config,
        reflection_client=reflection_client,
        dataset_builder_factory=dataset_builder_factory,
        sampling_client=sampling_client,
        log_path=log_path,
    )

    best_candidate = await optimizer.optimize()

    # Create strategy from best prompt
    strategy = SystemPromptStrategy(
        system_prompt=best_candidate.prompt_text,
        strategy_name=f"gepa_{best_candidate.candidate_id}",
    )

    print(f"\n  Best prompt: {best_candidate.prompt_text[:100]}...")
    print(f"  Best score: {best_candidate.best_score:.1%}")

    return strategy, best_candidate


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

    This is the θ optimization step in (s, θ) block coordinate ascent.
    """
    print("\n" + "=" * 60)
    print(f"RL PHASE: Training θ with optimized prompt")
    print("=" * 60)
    print(f"  Strategy: {strategy.name}")
    print(f"  Batches: {config.rl_batches_per_round}")
    print(f"  Starting θ from: {state_path or 'base model'}")

    # Build RL config with the strategy
    rl_config = Config(
        learning_rate=config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=config.env,
            batch_size=config.rl_groups_per_batch,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.rl_group_size,
            seed=config.seed,  # Use same seed across rounds for fair comparison
            prompt_template="none",
            n_batches=config.rl_batches_per_round,
            strategy=strategy,
        ),
        model_name=config.model_name,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        wandb_project=config.wandb_project,
        wandb_name=None,
        log_path=round_log_path,
        base_url=config.base_url,
        load_checkpoint_path=state_path,
        eval_every=max(1, config.rl_batches_per_round // 2),
        save_every=config.rl_batches_per_round,
    )

    # Run training
    await rl_main(rl_config)

    # Get checkpoint paths
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
    print("GEPA+RL ALTERNATION: Block Coordinate Ascent on (s, θ)")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Environment: {config.env}")
    print(f"Rounds: {config.n_rounds}")
    print(f"GEPA iterations per round: {config.gepa_iterations}")
    print(f"RL batches per round: {config.rl_batches_per_round}")
    print(f"Reflection model: {config.reflection_backend}/{config.reflection_model}")
    print("=" * 70)

    # Setup
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Determine base log path
    if config.log_path:
        base_log_path = config.log_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wrapper_root = Path(__file__).parent.parent
        base_log_path = str(wrapper_root / "runs" / f"alternation-{timestamp}")

    base_log_path = os.path.expanduser(base_log_path)
    os.makedirs(base_log_path, exist_ok=True)

    # Initialize wandb
    wandb_run = None
    if config.wandb_project and wandb is not None:
        # Determine mode for easy comparison
        mode = "gepa_rl" if config.gepa_iterations > 0 else "rl_only"
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=os.path.basename(base_log_path),
            tags=[mode, config.env],
            config={
                "mode": mode,
                "model_name": config.model_name,
                "env": config.env,
                "n_rounds": config.n_rounds,
                "gepa_iterations": config.gepa_iterations,
                "skip_first_gepa": config.skip_first_gepa,
                "rl_batches_per_round": config.rl_batches_per_round,
                "rl_groups_per_batch": config.rl_groups_per_batch,
                "rl_group_size": config.rl_group_size,
                "total_rl_batches": config.n_rounds * config.rl_batches_per_round,
                "reflection_model": f"{config.reflection_backend}/{config.reflection_model}" if config.gepa_iterations > 0 else "none",
            },
            dir=base_log_path,
        )

    # State
    current_paths: CheckpointPaths | None = None
    current_best_prompt: str | None = None
    results: list[RoundResult] = []

    for round_idx in range(config.n_rounds):
        print(f"\n{'#' * 70}")
        print(f"# ROUND {round_idx + 1} / {config.n_rounds}")
        print(f"{'#' * 70}")

        # 1. GEPA phase: optimize s with θ fixed
        # Skip GEPA in first round if configured (to let RL train first)
        skip_gepa_this_round = (round_idx == 0 and config.skip_first_gepa)

        sampler_path = current_paths.sampler_path if current_paths else None
        gepa_log_path = os.path.join(base_log_path, f"round{round_idx}_gepa")
        os.makedirs(gepa_log_path, exist_ok=True)

        if skip_gepa_this_round:
            print("\n" + "=" * 60)
            print("GEPA PHASE: Skipped (skip_first_gepa=True)")
            print("=" * 60)
            # Use default seed prompt
            gepa_candidate = GEPACandidate(
                prompt_text=DEFAULT_SEED_PROMPT,
                metadata=GEPACandidateMetadata(
                    candidate_id="seed",
                    parent_id=None,
                    iteration_created=0,
                    proposal_reasoning="Default seed prompt (GEPA skipped)",
                ),
                scores=[0.0],
                best_score=0.0,
                avg_score=0.0,
            )
            strategy = SystemPromptStrategy(
                system_prompt=DEFAULT_SEED_PROMPT,
                strategy_name="seed_prompt",
            )
        else:
            strategy, gepa_candidate = await run_gepa_phase(
                config=config,
                sampler_path=sampler_path,
                renderer_name=renderer_name,
                log_path=gepa_log_path,
                seed_prompt=current_best_prompt,
            )

        # Update best prompt for next round
        current_best_prompt = gepa_candidate.prompt_text

        # Log GEPA results to wandb
        if wandb_run is not None and wandb is not None:
            wandb.log({
                "round": round_idx,
                "gepa/best_score": gepa_candidate.best_score,
                "gepa/iterations": gepa_candidate.metadata.iteration_created,
                "gepa/skipped": skip_gepa_this_round,
            })

        # 2. RL phase: optimize θ with s fixed
        state_path = current_paths.state_path if current_paths else None
        rl_log_path = os.path.join(base_log_path, f"round{round_idx}_rl")

        new_paths = await run_rl_phase(
            config=config,
            strategy=strategy,
            round_idx=round_idx,
            state_path=state_path,
            renderer_name=renderer_name,
            round_log_path=rl_log_path,
        )

        # Record results
        result = RoundResult(
            round_idx=round_idx,
            gepa_best_prompt=gepa_candidate.prompt_text,
            gepa_best_score=gepa_candidate.best_score,
            gepa_iterations_run=gepa_candidate.metadata.iteration_created,
            rl_final_correct_rate=None,
            checkpoint_paths=new_paths,
        )
        results.append(result)

        # Update state
        current_paths = new_paths

        # Save round summary
        summary_path = os.path.join(base_log_path, "alternation_log.jsonl")
        with open(summary_path, "a") as f:
            f.write(
                json.dumps({
                    "round": round_idx,
                    "gepa_best_prompt": gepa_candidate.prompt_text,
                    "gepa_best_score": gepa_candidate.best_score,
                    "gepa_candidate_id": gepa_candidate.candidate_id,
                    "state_path": new_paths.state_path if new_paths else None,
                    "sampler_path": new_paths.sampler_path if new_paths else None,
                })
                + "\n"
            )

    # Close wandb
    if wandb_run is not None and wandb is not None:
        wandb.finish()

    return results


def print_summary(results: list[RoundResult]) -> None:
    """Print final summary."""
    print("\n" + "=" * 70)
    print("ALTERNATION SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\nRound {r.round_idx}:")
        print(f"  GEPA best score: {r.gepa_best_score:.1%}")
        print(f"  GEPA iterations: {r.gepa_iterations_run}")
        print(f"  Best prompt: {r.gepa_best_prompt[:80]}...")
        if r.checkpoint_paths:
            print(f"  θ sampler: {r.checkpoint_paths.sampler_path}")

    # Track prompt evolution
    prompts = [r.gepa_best_prompt[:50] + "..." for r in results]
    print(f"\nPrompt evolution:")
    for i, p in enumerate(prompts):
        print(f"  Round {i}: {p}")

    print("=" * 70)


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(AlternationConfig)
    results = asyncio.run(run_alternation(config))
    print_summary(results)


if __name__ == "__main__":
    main()
