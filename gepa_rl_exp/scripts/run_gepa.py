"""
GEPA: Trace-based reflective prompt optimization.

Implements the GEPA algorithm (arxiv:2507.19457) for optimizing prompts
by analyzing execution traces and using LLM reflection to propose
targeted improvements.

This replaces the old template-selection approach with true GEPA:
1. Collect execution traces with current prompt
2. Reflection LM analyzes traces to diagnose failures
3. Reflection LM proposes targeted prompt improvement
4. Evaluate and iterate

Usage:
    cd gepa_rl_exp
    python -m scripts.run_gepa env=arithmetic max_iterations=5
    python -m scripts.run_gepa env=gsm8k reflection_model=gpt-4o
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
from tinker_cookbook import model_info
from tinker_cookbook.rl.metric_util import dataset_to_env_group_builders

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gepa import GEPAConfig, GEPAOptimizer, GEPACandidate
from src.gepa.reflection import ReflectionClient, DummyReflectionClient
from src.train import get_dataset_builder
from src.prompt_strategy import SystemPromptStrategy


@chz.chz
class RunGEPAConfig:
    """Configuration for running GEPA optimization."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    checkpoint_path: str | None = None  # If None, use base model
    renderer_name: str | None = None

    # Environment
    env: Literal["arithmetic", "gsm8k", "math"] = "arithmetic"
    seed: int = 0

    # GEPA settings
    max_iterations: int = 5  # Number of GEPA optimization steps
    traces_per_iteration: int = 16  # Problems per iteration
    rollouts_per_problem: int = 4  # For variance estimation
    pool_size: int = 5  # Max candidates in pool
    min_improvement_threshold: float = 0.01

    # Reflection model
    reflection_backend: Literal["openai", "anthropic"] = "anthropic"
    reflection_model: str = "claude-opus-4-5-20251101"
    reflection_temperature: float = 0.7

    # Seed prompt (starting point for optimization)
    seed_prompt: str | None = None

    # Sampling settings
    max_tokens: int = 256
    temperature: float = 1.0

    # Output
    log_path: str | None = None
    log_traces: bool = True
    log_proposals: bool = True
    wandb_project: str | None = "gepa-rl"  # Default wandb project

    # Debug mode (uses dummy reflection client)
    debug: bool = False


def create_dataset_builder_factory(
    config: RunGEPAConfig,
    renderer_name: str,
):
    """
    Create a factory function that builds datasets with a given prompt.

    The factory takes a prompt string and returns a dataset builder
    configured with that prompt as the system prompt.
    """

    def factory(prompt: str):
        # Create a strategy with the given prompt
        strategy = SystemPromptStrategy(
            system_prompt=prompt,
            strategy_name="gepa_candidate",
        )

        return get_dataset_builder(
            env=config.env,
            batch_size=1,
            model_name=config.model_name,
            renderer_name=renderer_name,
            group_size=config.rollouts_per_problem,
            seed=config.seed,
            prompt_template="none",  # Use strategy instead
            n_batches=config.traces_per_iteration,
            strategy=strategy,
        )

    return factory


async def run_gepa(config: RunGEPAConfig) -> GEPACandidate:
    """
    Run GEPA optimization.

    Uses trace-based reflection to iteratively improve prompts.
    """
    print("=" * 60)
    print("GEPA: Trace-Based Reflective Prompt Optimization")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Checkpoint: {config.checkpoint_path or 'base model'}")
    print(f"Environment: {config.env}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Reflection model: {config.reflection_backend}/{config.reflection_model}")
    print("=" * 60)

    # Setup log path
    if config.log_path:
        log_path = config.log_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wrapper_root = Path(__file__).parent.parent
        log_path = str(wrapper_root / "runs" / f"gepa-{timestamp}")

    log_path = os.path.expanduser(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Log path: {log_path}")

    # Initialize wandb
    wandb_run = None
    if config.wandb_project and wandb is not None:
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=os.path.basename(log_path),
            config={
                "model_name": config.model_name,
                "env": config.env,
                "max_iterations": config.max_iterations,
                "traces_per_iteration": config.traces_per_iteration,
                "reflection_model": f"{config.reflection_backend}/{config.reflection_model}",
            },
            dir=log_path,
        )
        print(f"Wandb: {config.wandb_project}")

    # Auto-detect renderer
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    print(f"Renderer: {renderer_name}")

    # Create reflection client
    if config.debug:
        print("\n[DEBUG MODE] Using dummy reflection client")
        reflection_client = DummyReflectionClient()
    else:
        reflection_client = ReflectionClient(
            backend=config.reflection_backend,
            model=config.reflection_model,
            temperature=config.reflection_temperature,
        )

    # Create service client
    service = tinker.ServiceClient()

    # Create sampling client
    if config.checkpoint_path:
        print(f"\nLoading checkpoint: {config.checkpoint_path}")
        sampling_client = service.create_sampling_client(
            model_path=config.checkpoint_path,
            base_model=config.model_name,
        )
    else:
        print("\nUsing base model")
        sampling_client = service.create_sampling_client(base_model=config.model_name)

    # Create dataset builder factory
    dataset_builder_factory = create_dataset_builder_factory(config, renderer_name)

    # Create GEPA config
    gepa_config = GEPAConfig(
        reflection_backend=config.reflection_backend,
        reflection_model=config.reflection_model,
        reflection_temperature=config.reflection_temperature,
        max_iterations=config.max_iterations,
        traces_per_iteration=config.traces_per_iteration,
        rollouts_per_problem=config.rollouts_per_problem,
        pool_size=config.pool_size,
        min_improvement_threshold=config.min_improvement_threshold,
        seed_prompt=config.seed_prompt,
        log_traces=config.log_traces,
        log_proposals=config.log_proposals,
        seed=config.seed,
    )

    # Create optimizer
    optimizer = GEPAOptimizer(
        config=gepa_config,
        reflection_client=reflection_client,
        dataset_builder_factory=dataset_builder_factory,
        sampling_client=sampling_client,
        log_path=log_path,
    )

    # Run optimization
    best_candidate = await optimizer.optimize()

    # Save final results
    results = {
        "best_prompt": best_candidate.prompt_text,
        "best_score": best_candidate.best_score,
        "avg_score": best_candidate.avg_score,
        "candidate_id": best_candidate.candidate_id,
        "metadata": best_candidate.metadata.to_dict(),
        "total_iterations": optimizer.iteration,
        "config": {
            "model_name": config.model_name,
            "env": config.env,
            "max_iterations": config.max_iterations,
            "reflection_model": f"{config.reflection_backend}/{config.reflection_model}",
        },
    }

    results_path = os.path.join(log_path, "gepa_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save optimizer state for potential resume
    state_path = os.path.join(log_path, "gepa_state.json")
    with open(state_path, "w") as f:
        json.dump(optimizer.get_state(), f, indent=2)

    # Log final results to wandb
    if wandb_run is not None and wandb is not None:
        wandb.log({
            "final/best_score": best_candidate.best_score,
            "final/avg_score": best_candidate.avg_score,
            "final/iterations": optimizer.iteration,
        })
        wandb.finish()

    return best_candidate


def print_results(candidate: GEPACandidate) -> None:
    """Print final results."""
    print("\n" + "=" * 60)
    print("GEPA OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nBest prompt:")
    print("-" * 40)
    print(candidate.prompt_text)
    print("-" * 40)
    print(f"\nBest score: {candidate.best_score:.1%}")
    print(f"Average score: {candidate.avg_score:.1%}")
    print(f"Candidate ID: {candidate.candidate_id}")
    if candidate.metadata.failure_patterns:
        print(f"Addressed patterns: {candidate.metadata.failure_patterns}")
    print("=" * 60)


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(RunGEPAConfig)
    best = asyncio.run(run_gepa(config))
    print_results(best)


if __name__ == "__main__":
    main()
