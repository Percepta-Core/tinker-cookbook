"""
GEPA Phase 1: Prompt template comparison on fixed weights.

Evaluates different prompt templates by running rollouts and comparing
mean reward and format success rate. Used to select the best template
before (or during) RL training.

Usage:
    cd gepa_rl_exp
    python -m scripts.run_gepa env=arithmetic
    python -m scripts.run_gepa env=gsm8k num_problems=50
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics, dataset_to_env_group_builders
from tinker_cookbook.rl.rollouts import do_group_rollout

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.train import get_dataset_builder


@chz.chz
class GEPAConfig:
    """Configuration for GEPA prompt evaluation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    checkpoint_path: str | None = None  # If None, use base model (θ_0)
    renderer_name: str | None = None

    # Environment
    env: Literal["arithmetic", "gsm8k", "math"] = "arithmetic"
    seed: int = 0

    # Evaluation settings
    num_problems: int = 32  # Number of problems to evaluate per template
    group_size: int = 4  # Rollouts per problem (for variance estimation)
    max_tokens: int = 128
    temperature: float = 1.0

    # Templates to compare
    templates: str = "baseline,step_by_step,concise"  # Comma-separated

    # Output
    output_path: str | None = None  # If None, print to stdout


@dataclass
class TemplateResult:
    """Results for a single template evaluation."""

    template: str
    mean_reward: float
    std_reward: float
    format_success_rate: float
    correct_rate: float
    num_episodes: int
    avg_tokens_per_episode: float


async def evaluate_template(
    sampling_client: tinker.SamplingClient,
    template: str,
    config: GEPAConfig,
    renderer_name: str,
) -> TemplateResult:
    """Evaluate a single template by running rollouts."""

    print(f"\n  Evaluating template: {template}")

    # Create dataset builder with this template
    dataset_builder = get_dataset_builder(
        env=config.env,
        batch_size=1,  # One group at a time
        model_name=config.model_name,
        renderer_name=renderer_name,
        group_size=config.group_size,
        seed=config.seed,
        prompt_template=template,
        n_batches=config.num_problems,  # For arithmetic env
    )

    # Build the dataset (async) - returns (train_dataset, test_dataset)
    train_dataset, _ = await dataset_builder()

    # Get env group builders from dataset
    all_env_builders = dataset_to_env_group_builders(train_dataset)
    env_builders = all_env_builders[: config.num_problems]

    print(f"    Evaluating {len(env_builders)} problems...")

    # Create policy from sampling client
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Run rollouts for all problems
    trajectory_groups = await asyncio.gather(
        *[do_group_rollout(builder, policy) for builder in env_builders]
    )

    # Collect results
    all_rewards: list[float] = []
    all_correct: list[float] = []
    all_format_ok: list[float] = []
    all_tokens: list[int] = []

    for trajectory_group in trajectory_groups:
        # Extract metrics from each trajectory in the group
        for traj in trajectory_group.trajectories_G:
            # Total reward for this episode
            episode_reward = sum(t.reward for t in traj.transitions)
            all_rewards.append(episode_reward)

            # Check for correct/format metrics in transitions
            for t in traj.transitions:
                if "correct" in t.metrics:
                    all_correct.append(float(t.metrics["correct"]))
                if "format" in t.metrics:
                    all_format_ok.append(float(t.metrics["format"]))

            # Token count (ac.tokens is the token list)
            total_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
            all_tokens.append(total_tokens)

    # Compute statistics
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    std_reward = (
        (sum((r - mean_reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5
        if len(all_rewards) > 1
        else 0.0
    )
    format_rate = sum(all_format_ok) / len(all_format_ok) if all_format_ok else 1.0
    correct_rate = sum(all_correct) / len(all_correct) if all_correct else 0.0
    avg_tokens = sum(all_tokens) / len(all_tokens) if all_tokens else 0.0

    return TemplateResult(
        template=template,
        mean_reward=mean_reward,
        std_reward=std_reward,
        format_success_rate=format_rate,
        correct_rate=correct_rate,
        num_episodes=len(all_rewards),
        avg_tokens_per_episode=avg_tokens,
    )


async def run_gepa(config: GEPAConfig) -> list[TemplateResult]:
    """Run GEPA evaluation across all templates."""

    print("=" * 60)
    print("GEPA Phase 1: Prompt Template Comparison")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Checkpoint: {config.checkpoint_path or 'base model (θ_0)'}")
    print(f"Environment: {config.env}")
    print(f"Problems per template: {config.num_problems}")
    print(f"Rollouts per problem: {config.group_size}")
    print("=" * 60)

    # Parse templates
    templates = [t.strip() for t in config.templates.split(",")]
    print(f"Templates to evaluate: {templates}")

    # Auto-detect renderer
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )
    print(f"Renderer: {renderer_name}")

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
        print("\nUsing base model (θ_0)")
        sampling_client = service.create_sampling_client(base_model=config.model_name)

    # Evaluate each template
    results: list[TemplateResult] = []
    for template in templates:
        result = await evaluate_template(
            sampling_client=sampling_client,
            template=template,
            config=config,
            renderer_name=renderer_name,
        )
        results.append(result)

    return results


def print_results(results: list[TemplateResult]) -> str:
    """Print results table and return winner."""

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Header
    print(
        f"{'Template':<15} {'Reward':>10} {'Std':>8} {'Correct':>10} {'Format':>10} {'Tokens':>8}"
    )
    print("-" * 60)

    # Sort by mean reward (descending)
    sorted_results = sorted(results, key=lambda r: r.mean_reward, reverse=True)

    for r in sorted_results:
        print(
            f"{r.template:<15} {r.mean_reward:>10.3f} {r.std_reward:>8.3f} "
            f"{r.correct_rate:>9.1%} {r.format_success_rate:>9.1%} {r.avg_tokens_per_episode:>8.1f}"
        )

    winner = sorted_results[0]
    print("-" * 60)
    print(f"WINNER: {winner.template} (reward={winner.mean_reward:.3f})")
    print("=" * 60)

    return winner.template


def save_results(results: list[TemplateResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "template": r.template,
                "mean_reward": r.mean_reward,
                "std_reward": r.std_reward,
                "format_success_rate": r.format_success_rate,
                "correct_rate": r.correct_rate,
                "num_episodes": r.num_episodes,
                "avg_tokens_per_episode": r.avg_tokens_per_episode,
            }
            for r in results
        ],
        "winner": max(results, key=lambda r: r.mean_reward).template,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(GEPAConfig)
    results = asyncio.run(run_gepa(config))

    winner = print_results(results)

    if config.output_path:
        save_results(results, config.output_path)

    print(f"\nRecommendation: Use template '{winner}' for RL training")


if __name__ == "__main__":
    main()
