"""
Expert Iteration with Teacher Hints.

Implements the teacher-hint-based expert iteration pipeline:
1. For each problem x, run student → get initial response y0
2. Teacher generates hint h (without revealing answer)
3. Run student with hint → sample N candidates
4. Select y* = argmax R(x, y)
5. Distill: train θ on (x → y*) with CLEAN harness (no hint)

Key invariants:
- Student NEVER sees hint during distillation
- Expert refinement DOES use hint
- Selection uses existing R(x, y) via check_answer()

Usage:
    cd gepa_rl_exp
    python -m scripts.run_expert_iteration env=arithmetic n_batches=5
    python -m scripts.run_expert_iteration env=gsm8k teacher_model=gpt-4o-mini
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

import chz
import tinker
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.metric_util import dataset_to_env_group_builders
from tinker_cookbook.rl.problem_env import ProblemEnv
from tinker_cookbook.rl.rollouts import do_group_rollout
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils import logtree, ml_log

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.expert_iteration import (
    ExpertIterationConfig,
    ExpertIterationProcessor,
    ExpertIterationResult,
    create_distillation_datum,
    process_batch_expert_iteration,
)
from src.expert_iteration_logging import (
    ExpertIterationLogger,
    compute_batch_metrics,
)
from src.hint_strategy import create_clean_strategy
from src.prompt_templates import get_strategy_by_name
from src.teacher_client import (
    DummyTeacherClient,
    TeacherClient,
    TeacherConfig,
    create_teacher_client,
)
from src.train import get_dataset_builder


@chz.chz
class ExpertIterationTrainingConfig:
    """Configuration for expert iteration training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Environment configuration
    env: Literal["gsm8k", "math", "arithmetic"] = "arithmetic"
    seed: int = 0
    n_batches: int = 10

    # Prompt strategy (base harness for distillation)
    prompt_template: Literal["baseline", "step_by_step", "concise", "none"] = "baseline"

    # Expert iteration settings
    enable_teacher_hints: bool = True
    teacher_backend: Literal["openai", "anthropic", "dummy"] = "openai"
    teacher_model: str = "gpt-4o-mini"
    hint_max_chars: int = 500
    expert_samples_per_x: int = 8  # N candidates with hint
    hint_filtering_strictness: Literal["strict", "moderate", "lenient"] = "moderate"
    fraction_of_batch_with_hints: float = 1.0

    # Training hyperparameters
    groups_per_batch: int = 32  # Problems per batch
    learning_rate: float = 1e-5
    max_tokens: int = 256
    temperature: float = 1.0

    # Loss function for distillation
    loss_fn: Literal["importance_sampling", "cross_entropy"] = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Service configuration
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Checkpointing
    save_every: int = 5


def create_teacher(config: ExpertIterationTrainingConfig) -> TeacherClient | None:
    """Create teacher client based on config."""
    if not config.enable_teacher_hints:
        return None

    if config.teacher_backend == "dummy":
        return DummyTeacherClient()

    teacher_config = TeacherConfig(
        backend=config.teacher_backend,  # type: ignore
        model=config.teacher_model,
        max_hint_chars=config.hint_max_chars,
        temperature=0.7,
    )
    return create_teacher_client(teacher_config)


async def run_expert_iteration_training(config: ExpertIterationTrainingConfig) -> None:
    """Main training loop for expert iteration."""

    # Setup
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        config.model_name
    )

    # Generate run name
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"expert-iter-{config.env}-{model_slug}-{timestamp}"

    # Determine log path
    if config.log_path:
        log_path = config.log_path
    else:
        wrapper_root = Path(__file__).parent.parent
        log_path = str(wrapper_root / "runs" / run_name)
    log_path = os.path.expanduser(log_path)

    # Check log directory
    cli_utils.check_log_dir(log_path, behavior_if_exists=config.behavior_if_log_dir_exists)
    os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    ml_logger = ml_log.setup_logging(
        log_dir=log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=config.wandb_name or run_name,
    )
    expert_logger = ExpertIterationLogger(log_path=log_path)

    print("=" * 70)
    print("EXPERT ITERATION TRAINING")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Environment: {config.env}")
    print(f"Teacher: {config.teacher_backend}/{config.teacher_model}")
    print(f"Expert samples per problem: {config.expert_samples_per_x}")
    print(f"Problems per batch: {config.groups_per_batch}")
    print(f"Batches: {config.n_batches}")
    print(f"Log path: {log_path}")
    print("=" * 70)

    # Create service and clients
    service = tinker.ServiceClient(base_url=config.base_url)

    # Create training client
    if config.load_checkpoint_path:
        # Starting from checkpoint - load weights only (fresh optimizer)
        training_client = await service.create_training_client_from_state_async(
            config.load_checkpoint_path
        )
        print(f"Loaded checkpoint from: {config.load_checkpoint_path}")
    else:
        training_client = await service.create_lora_training_client_async(
            config.model_name, rank=config.lora_rank
        )

    # Get tokenizer from training client
    tokenizer = training_client.get_tokenizer()

    # Create sampling client
    sampling_client = service.create_sampling_client(base_model=config.model_name)

    # Create teacher client
    teacher_client = create_teacher(config)

    # Get renderer using tokenizer
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # Get base strategy for distillation (clean, no hint)
    base_strategy = get_strategy_by_name(config.prompt_template)
    clean_strategy = create_clean_strategy(base_strategy)

    # Expert iteration config
    expert_config = ExpertIterationConfig(
        enable_teacher_hints=config.enable_teacher_hints,
        teacher_backend=config.teacher_backend if config.teacher_backend != "dummy" else "openai",
        teacher_model=config.teacher_model,
        hint_max_chars=config.hint_max_chars,
        expert_samples_per_x=config.expert_samples_per_x,
        hint_filtering_strictness=config.hint_filtering_strictness,
        fraction_of_batch_with_hints=config.fraction_of_batch_with_hints,
    )

    # Create dataset
    dataset_builder = get_dataset_builder(
        env=config.env,
        batch_size=config.groups_per_batch,
        model_name=config.model_name,
        renderer_name=renderer_name,
        group_size=1,  # Single rollout for initial y0
        seed=config.seed,
        prompt_template=config.prompt_template,
        n_batches=config.n_batches,
        strategy=base_strategy,
    )

    train_dataset, _ = await dataset_builder()
    all_env_builders = dataset_to_env_group_builders(train_dataset)

    # Create policy for rollouts
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Create expert iteration processor
    processor = ExpertIterationProcessor(
        config=expert_config,
        sampling_client=sampling_client,
        renderer=renderer,
        teacher_client=teacher_client,
        base_strategy=base_strategy,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Training loop
    batch_idx = 0
    for batch_start in range(0, len(all_env_builders), config.groups_per_batch):
        batch_builders = all_env_builders[batch_start : batch_start + config.groups_per_batch]

        print(f"\n--- Batch {batch_idx + 1}/{config.n_batches} ---")

        with logtree.scope_header(f"Batch {batch_idx}"):
            # Step 1: Run initial rollouts to get y0
            print("  Running initial rollouts...")
            trajectory_groups: list[TrajectoryGroup] = await asyncio.gather(
                *[do_group_rollout(builder, policy) for builder in batch_builders]
            )

            # Extract envs and trajectories
            # Each group has 1 trajectory since group_size=1
            envs: list[ProblemEnv] = []
            trajectories: list[Trajectory] = []
            for builder, traj_group in zip(batch_builders, trajectory_groups):
                env_list = await builder.make_envs()
                # Use first (and only) env and trajectory
                envs.append(env_list[0])  # type: ignore
                trajectories.append(traj_group.trajectories_G[0])

            # Step 2: Process through expert iteration
            print("  Processing expert iteration (hints + refinement)...")
            results: list[ExpertIterationResult] = await process_batch_expert_iteration(
                envs=envs,
                trajectories=trajectories,
                processor=processor,
            )

            # Step 3: Compute and log metrics
            batch_metrics = compute_batch_metrics(results)
            batch_metrics["progress/batch"] = batch_idx

            # Log some examples
            expert_logger.log_batch(results, batch_idx)
            expert_logger.save()

            # Log metrics
            ml_logger.log_metrics(batch_metrics, step=batch_idx)

            print(f"  Initial mean reward: {batch_metrics.get('expert_iter/mean_initial_reward', 0):.3f}")
            print(f"  Best mean reward: {batch_metrics.get('expert_iter/mean_best_reward', 0):.3f}")
            print(f"  Improvement rate: {batch_metrics.get('expert_iter/improvement_rate', 0):.1%}")

            # Step 4: Create distillation datums
            print("  Creating distillation datums...")
            datums: list[tinker.Datum] = []
            for result in results:
                # Only create datums for successful improvements
                if result.best_reward > 0:
                    datum = create_distillation_datum(
                        question=result.question,
                        best_response=result.best_candidate,
                        reward=result.best_reward,
                        renderer=renderer,
                        clean_strategy=clean_strategy,
                        max_length=2048,
                    )
                    datums.append(datum)

            if not datums:
                print("  No positive-reward datums, skipping training step")
                batch_idx += 1
                continue

            # Step 5: Train on distillation datums
            print(f"  Training on {len(datums)} datums...")

            # Create adam params
            adam_params = tinker.AdamParams(
                learning_rate=config.learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
            )

            # Pipeline forward_backward and optim_step
            train_fut = await training_client.forward_backward_async(
                data=datums,
                loss_fn=config.loss_fn,
            )
            optim_fut = await training_client.optim_step_async(adam_params)

            # Await results
            _ = await train_fut.result_async()
            await optim_fut.result_async()

            # Log training metrics
            train_metrics = {
                "train/n_datums": len(datums),
            }
            ml_logger.log_metrics(train_metrics, step=batch_idx)
            print(f"  Trained on {len(datums)} datums")

            # Step 6: Checkpoint
            if (batch_idx + 1) % config.save_every == 0:
                print("  Saving checkpoint...")
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    log_path=log_path,
                    batch_idx=batch_idx,
                )

        batch_idx += 1
        if batch_idx >= config.n_batches:
            break

    # Final checkpoint
    print("\nSaving final checkpoint...")
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        log_path=log_path,
        batch_idx=batch_idx,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Log path: {log_path}")


def main() -> None:
    """Entry point."""
    config = chz.entrypoint(ExpertIterationTrainingConfig)
    asyncio.run(run_expert_iteration_training(config))


if __name__ == "__main__":
    main()
