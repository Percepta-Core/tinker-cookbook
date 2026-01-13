import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Protocol, Sequence, runtime_checkable

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.utils import logtree

logger = logging.getLogger(__name__)


# =============================================================================
# PromptStrategy Protocol - first-class prompt/harness abstraction
# =============================================================================


@runtime_checkable
class PromptStrategy(Protocol):
    """
    First-class prompt/harness object `s` for (s, θ) optimization.

    This protocol is dataset-agnostic: it makes no assumptions about
    answer formats (LaTeX, boxed, delimiters, etc.). The reward function
    R(x, y) remains external to the strategy.

    The strategy controls:
    - How to build the message list from a question
    - Optional output postprocessing before reward computation
    """

    def build_messages(self, question: str) -> list[renderers.Message]:
        """Build the full message list for a question."""
        ...

    def postprocess_output(self, raw_output: str) -> str:
        """Optional output postprocessing before reward computation."""
        ...

    @property
    def name(self) -> str:
        """Unique identifier for this strategy."""
        ...


class ProblemEnv(Env):
    """
    Base class for problem-solving environments.

    Supports two modes for prompt construction:
    1. Legacy mode: Use `convo_prefix` (list of few-shot messages)
    2. Strategy mode: Use `strategy` (PromptStrategy object)

    If both are provided, `strategy` takes precedence.
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        format_coef: float = 0.1,
        strategy: PromptStrategy | None = None,
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef
        self._strategy = strategy

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    @abstractmethod
    def get_question(self) -> str:
        pass

    @abstractmethod
    def check_answer(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def check_format(self, sample_str: str) -> bool:
        pass

    @abstractmethod
    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        pass

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        question = self.get_question()
        if self._strategy is not None:
            # Strategy mode: use strategy to build messages
            convo = self._strategy.build_messages(question)
        else:
            # Legacy mode: use convo_prefix
            convo = self.convo_prefix + [{"role": "user", "content": question}]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        raw_content = renderers.get_text_content(message)

        # Apply strategy postprocessing if available
        if self._strategy is not None:
            content = self._strategy.postprocess_output(raw_content)
        else:
            content = raw_content

        correct_format = float(parse_success) and float(self.check_format(content))
        correct_answer = float(self.check_answer(content))
        total_reward = self.format_coef * (correct_format - 1) + correct_answer

        # Log the attempt
        logtree.log_text(f"Problem: {self.get_question()}")
        logtree.log_text(f"Response: {message['content']}")
        if content != raw_content:
            logtree.log_text(f"Postprocessed: {content}")
        logtree.log_text(f"Reference Answer: {self.get_reference_answer()}")
        logtree.log_text(
            f"Format Valid: {'✓' if correct_format else '✗'}, Correct: {'✓' if correct_answer else '✗'}, Reward: {total_reward:.2f}"
        )

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": correct_format,
                "correct": correct_answer,
            },
        )


@dataclass(frozen=True)
class ProblemGroupBuilder(EnvGroupBuilder):
    env_thunk: Callable[[], ProblemEnv]
    num_envs: int
    dataset_name: str = "problems"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]
