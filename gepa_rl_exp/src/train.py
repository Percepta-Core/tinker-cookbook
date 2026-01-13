"""
Main entry point for math RL training with prompt template/strategy support.

This module extends the tinker-cookbook math RL recipe with:
- Customizable prompt strategies (first-class `s` object for (s, θ) optimization)
- Organized output directory structure
- Sensible defaults for common use cases

Usage:
    python -m src.train env=gsm8k prompt_template=step_by_step
"""

import asyncio
import os
from datetime import datetime
from typing import Literal

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.arithmetic_env import ArithmeticDatasetBuilder
from tinker_cookbook.recipes.math_rl.math_env import (
    Gsm8kDatasetBuilder,
    MathDatasetBuilder,
    MathEnv,
)
from tinker_cookbook.rl.problem_env import PromptStrategy
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder

from .prompt_strategy import FewShotStrategy, NoPromptStrategy
from .prompt_templates import get_strategy_by_name, get_template


@chz.chz
class MathRLConfig:
    """Extended command-line configuration for math RL training with prompt templates."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: Literal["gsm8k", "math", "arithmetic"] = "gsm8k"
    seed: int = 0
    n_batches: int | None = None  # For arithmetic env: limit number of batches

    # Prompt template configuration
    prompt_template: Literal["baseline", "step_by_step", "concise", "none"] = "baseline"

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 100
    learning_rate: float = 1e-5
    max_tokens: int = 256
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False

    # Evals and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Advanced training options
    max_steps_off_policy: int | None = None
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"


def get_dataset_builder(
    env: str,
    batch_size: int,
    model_name: str,
    renderer_name: str,
    group_size: int,
    seed: int,
    prompt_template: str,
    n_batches: int | None = None,
    strategy: PromptStrategy | None = None,
) -> RLDatasetBuilder:
    """Create dataset builder with custom prompt template or strategy.

    The `strategy` parameter takes precedence over `prompt_template` when provided.
    This enables the (s, θ) joint optimization where `s` is the prompt strategy.

    Args:
        env: Environment name ("gsm8k", "math", or "arithmetic")
        batch_size: Number of groups per batch (groups_per_batch)
        model_name: Model name for tokenizer
        renderer_name: Renderer name (e.g., "llama3", "qwen3")
        group_size: Number of rollouts per problem
        seed: Random seed for data shuffling
        prompt_template: Template name or "none" for no few-shot examples
        n_batches: Number of batches (for arithmetic env)
        strategy: First-class PromptStrategy object (takes precedence over prompt_template)

    Returns:
        Configured RLDatasetBuilder
    """
    # Strategy takes precedence over prompt_template
    if strategy is not None:
        # Use the provided strategy directly
        pass
    elif prompt_template == "none":
        strategy = NoPromptStrategy(strategy_name="none")
    elif prompt_template == "standard":
        # Use the default from MathEnv as a FewShotStrategy
        strategy = FewShotStrategy(
            fewshot_messages=MathEnv.standard_fewshot_prefix(),
            strategy_name="standard",
        )
    else:
        # Use custom template as strategy
        strategy = get_strategy_by_name(prompt_template)

    if env == "arithmetic":
        return ArithmeticDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            n_batches=n_batches or 100,
            include_fewshot=False,  # Strategy handles this
            strategy=strategy,
        )
    elif env == "gsm8k":
        return Gsm8kDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            convo_prefix=None,  # Strategy handles this
            seed=seed,
            strategy=strategy,
        )
    elif env == "math":
        return MathDatasetBuilder(
            batch_size=batch_size,
            model_name_for_tokenizer=model_name,
            renderer_name=renderer_name,
            group_size=group_size,
            convo_prefix=None,  # Strategy handles this
            seed=seed,
            strategy=strategy,
        )
    else:
        raise ValueError(f"Unknown environment: {env}. Supported: arithmetic, gsm8k, math")


async def run_training(cli_config: MathRLConfig) -> None:
    """Convert CLI config to full config and run training.

    Args:
        cli_config: Parsed command-line configuration
    """
    # Auto-detect renderer if not specified
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Generate run name for logging
    model_name_slug = cli_config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{cli_config.env}-{model_name_slug}-{cli_config.prompt_template}-"
        f"lr{cli_config.learning_rate}-g{cli_config.group_size}x{cli_config.groups_per_batch}-"
        f"seed{cli_config.seed}-{timestamp}"
    )

    # Determine log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        # Default to runs/ directory relative to wrapper root
        wrapper_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_path = os.path.join(wrapper_root, "runs", run_name)

    # Expand user paths
    log_path = os.path.expanduser(log_path)

    # W&B run name defaults to auto-generated name
    wandb_name = cli_config.wandb_name or run_name

    # Build the full training config
    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            env=cli_config.env,
            batch_size=cli_config.groups_per_batch,
            model_name=cli_config.model_name,
            renderer_name=renderer_name,
            group_size=cli_config.group_size,
            seed=cli_config.seed,
            prompt_template=cli_config.prompt_template,
            n_batches=cli_config.n_batches,
        ),
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=(
            AsyncConfig(
                max_steps_off_policy=cli_config.max_steps_off_policy,
                groups_per_batch=cli_config.groups_per_batch,
            )
            if cli_config.max_steps_off_policy is not None
            else None
        ),
        loss_fn=cli_config.loss_fn,
    )

    # Check log directory handling
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    print(f"Starting training run: {run_name}")
    print(f"  Environment: {cli_config.env}")
    print(f"  Prompt template: {cli_config.prompt_template}")
    print(f"  Model: {cli_config.model_name}")
    print(f"  Log path: {log_path}")
    print()

    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(MathRLConfig)
    asyncio.run(run_training(cli_config))
