"""
Core expert iteration logic.

This module implements the main expert iteration pipeline:
1. Process problems through hint generation and expert refinement
2. Select best candidates based on reward
3. Create distillation datums for training

The key invariant is that distillation datums use the CLEAN strategy (no hint),
so the student learns to produce good answers without external guidance.
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Sequence

import tinker
import torch
from tinker import TensorData

from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.problem_env import ProblemEnv, PromptStrategy
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)

from .hint_generation import HintRequest, HintResult, generate_hint
from .hint_strategy import CleanStrategy, HintStrategy, create_clean_strategy, create_hint_strategy
from .prompt_strategy import FewShotStrategy

if TYPE_CHECKING:
    from .teacher_client import TeacherClient


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ExpertIterationConfig:
    """Configuration for expert iteration."""

    enable_teacher_hints: bool = True
    teacher_backend: Literal["openai", "anthropic"] = "openai"
    teacher_model: str = "gpt-4o"
    hint_max_chars: int = 500
    expert_samples_per_x: int = 8  # N samples with hint
    hint_filtering_strictness: Literal["strict", "moderate", "lenient"] = "moderate"
    fraction_of_batch_with_hints: float = 1.0  # For ablation: 0-1


# =============================================================================
# Results
# =============================================================================


@dataclass
class ExpertIterationResult:
    """Result from processing one problem with expert iteration."""

    question: str
    ground_truth: str
    initial_response: str  # y0
    initial_reward: float  # R(x, y0)
    hint: HintResult | None
    candidates: list[str]  # y1..yN
    candidate_rewards: list[float]
    best_candidate: str  # y*
    best_reward: float  # R(x, y*)
    used_hint: bool
    improvement: float = field(init=False)

    def __post_init__(self):
        self.improvement = self.best_reward - self.initial_reward


# =============================================================================
# Expert Iteration Processor
# =============================================================================


class ExpertIterationProcessor:
    """Processes problems through the expert iteration pipeline."""

    def __init__(
        self,
        config: ExpertIterationConfig,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        teacher_client: "TeacherClient | None",
        base_strategy: PromptStrategy,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ):
        self.config = config
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.teacher_client = teacher_client
        self.base_strategy = base_strategy
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create policy for sampling
        self.policy = TinkerTokenCompleter(
            sampling_client=sampling_client,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def process_problem(
        self,
        env: ProblemEnv,
        initial_trajectory: Trajectory,
    ) -> ExpertIterationResult:
        """
        Process a single problem through expert iteration.

        Args:
            env: The problem environment (contains question, ground truth, reward fn)
            initial_trajectory: The initial rollout result (y0)

        Returns:
            ExpertIterationResult with all processing information
        """
        # Extract information from env and trajectory
        question = env.get_question()
        ground_truth = env.get_reference_answer()

        # Get initial response from trajectory
        initial_response = self._extract_response_text(initial_trajectory)
        initial_reward = self._compute_trajectory_reward(initial_trajectory)

        # Decide whether to use hints for this problem
        use_hint = (
            self.config.enable_teacher_hints
            and self.teacher_client is not None
            and random.random() < self.config.fraction_of_batch_with_hints
        )

        hint_result: HintResult | None = None
        candidates: list[str] = []
        candidate_rewards: list[float] = []

        if use_hint:
            # Generate hint from teacher
            hint_request = HintRequest(
                question=question,
                student_response=initial_response,
                ground_truth=ground_truth,
            )
            assert self.teacher_client is not None  # Type narrowing
            hint_result = await generate_hint(
                client=self.teacher_client,
                request=hint_request,
                max_chars=self.config.hint_max_chars,
                safeguard_strictness=self.config.hint_filtering_strictness,
            )

            # Sample N candidates with hint
            candidates, candidate_rewards = await self._sample_with_hint(
                env=env,
                hint=hint_result.hint,
                n_samples=self.config.expert_samples_per_x,
            )
        else:
            # No hint - just use initial response
            candidates = [initial_response]
            candidate_rewards = [initial_reward]

        # Select best candidate
        best_idx = self._argmax(candidate_rewards)
        best_candidate = candidates[best_idx]
        best_reward = candidate_rewards[best_idx]

        return ExpertIterationResult(
            question=question,
            ground_truth=ground_truth,
            initial_response=initial_response,
            initial_reward=initial_reward,
            hint=hint_result,
            candidates=candidates,
            candidate_rewards=candidate_rewards,
            best_candidate=best_candidate,
            best_reward=best_reward,
            used_hint=use_hint,
        )

    async def _sample_with_hint(
        self,
        env: ProblemEnv,
        hint: str,
        n_samples: int,
    ) -> tuple[list[str], list[float]]:
        """
        Sample N candidates using hint-augmented strategy.

        Returns:
            (candidates, rewards) lists
        """
        # Create hint strategy
        hint_strategy = create_hint_strategy(self.base_strategy, hint)

        # Create a modified env that uses the hint strategy
        # We need to sample from the model with the hint in the prompt
        question = env.get_question()
        messages = hint_strategy.build_messages(question)
        prompt = self.renderer.build_generation_prompt(messages)
        stop_condition = self.renderer.get_stop_sequences()

        candidates: list[str] = []
        rewards: list[float] = []

        # Sample N times
        for _ in range(n_samples):
            # Sample from model
            ac_with_logprobs = await self.policy(prompt, stop_condition)

            # Parse response
            message, parse_success = self.renderer.parse_response(ac_with_logprobs.tokens)
            response_text = renderers.get_text_content(message)

            # Compute reward using env's check_answer
            correct = float(env.check_answer(response_text))
            format_ok = float(parse_success and env.check_format(response_text))
            reward = env.format_coef * (format_ok - 1) + correct

            candidates.append(response_text)
            rewards.append(reward)

        return candidates, rewards

    def _extract_response_text(self, trajectory: Trajectory) -> str:
        """Extract the response text from a trajectory."""
        if not trajectory.transitions:
            return ""
        # Get the action tokens from the first (and typically only) transition
        action_tokens = trajectory.transitions[0].ac.tokens
        message, _ = self.renderer.parse_response(action_tokens)
        return renderers.get_text_content(message)

    def _compute_trajectory_reward(self, trajectory: Trajectory) -> float:
        """Compute total reward from a trajectory."""
        return sum(t.reward for t in trajectory.transitions)

    def _argmax(self, values: list[float]) -> int:
        """Return index of maximum value."""
        if not values:
            return 0
        return max(range(len(values)), key=lambda i: values[i])


# =============================================================================
# Distillation Datum Creation
# =============================================================================


def create_distillation_datum(
    question: str,
    best_response: str,
    reward: float,
    renderer: renderers.Renderer,
    clean_strategy: PromptStrategy,
    max_length: int | None = None,
) -> tinker.Datum:
    """
    Create datum for RL-weighted distillation.

    CRITICAL: Uses clean_strategy (no hint) so student learns
    to produce y* without hint at inference time.

    The advantage is set to R(x, y*) for importance_sampling loss.

    Args:
        question: The problem question
        best_response: The best response y*
        reward: R(x, y*) - will be used as advantage
        renderer: Renderer for tokenization
        clean_strategy: Strategy WITHOUT hint for final prompt
        max_length: Optional max sequence length

    Returns:
        Datum ready for training with importance_sampling loss
    """
    # Build clean prompt (NO hint!)
    messages = clean_strategy.build_messages(question)

    # Add the best response as assistant turn
    full_conversation = messages + [{"role": "assistant", "content": best_response}]

    # Build model input using renderer
    model_input, weights = renderer.build_supervised_example(
        full_conversation,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # Truncate if needed
    if max_length is not None and model_input.length > max_length:
        # Truncate chunks
        chunks = list(model_input.chunks)
        total_len = 0
        truncated_chunks = []
        for chunk in chunks:
            if isinstance(chunk, tinker.EncodedTextChunk):
                remaining = max_length - total_len
                if remaining <= 0:
                    break
                if len(chunk.tokens) > remaining:
                    truncated_chunks.append(
                        tinker.EncodedTextChunk(tokens=chunk.tokens[:remaining])
                    )
                    break
                truncated_chunks.append(chunk)
                total_len += len(chunk.tokens)
            else:
                truncated_chunks.append(chunk)
                total_len += chunk.length
        model_input = tinker.ModelInput(chunks=truncated_chunks)
        weights = weights[: model_input.length]

    # Create right-shifted input and left-shifted targets
    input_tokens, target_tokens = create_rightshifted_model_input_and_leftshifted_targets(
        list(model_input.chunks)
    )

    # Adjust weights to match target length
    target_weights = weights[1:]  # Left shift weights too

    # Create advantages: reward for response tokens (where weight > 0)
    advantages = [reward if w > 0 else 0.0 for w in target_weights]

    # Create mask: 1.0 for response tokens, 0.0 for prompt
    mask = [1.0 if w > 0 else 0.0 for w in target_weights]

    # For off-policy distillation, set logprobs to 0
    # (we don't have the student's logprobs for y*)
    logprobs = [0.0] * len(target_tokens)

    return tinker.Datum(
        model_input=input_tokens,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
            "logprobs": TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
            "advantages": TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
            "mask": TensorData.from_torch(torch.tensor(mask, dtype=torch.float32)),
        },
    )


async def process_batch_expert_iteration(
    envs: Sequence[ProblemEnv],
    trajectories: Sequence[Trajectory],
    processor: ExpertIterationProcessor,
) -> list[ExpertIterationResult]:
    """
    Process a batch of problems through expert iteration.

    Args:
        envs: List of problem environments
        trajectories: List of initial rollout trajectories
        processor: ExpertIterationProcessor instance

    Returns:
        List of ExpertIterationResult for each problem
    """
    results = await asyncio.gather(
        *[
            processor.process_problem(env, traj)
            for env, traj in zip(envs, trajectories)
        ]
    )
    return list(results)
