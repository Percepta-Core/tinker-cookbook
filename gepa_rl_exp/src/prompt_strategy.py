"""
PromptStrategy: First-class object for prompt/harness abstraction.

This module defines the `s` in the (s, θ) joint optimization:
- s controls the prompt/harness (how to present problems)
- θ controls the model weights

The PromptStrategy is dataset-agnostic: it makes no assumptions about
answer formats (LaTeX, boxed, delimiters, etc.). Dataset-specific
behavior (question formatting, answer parsing) remains in the Env.

Usage:
    # Create a strategy
    strategy = FewShotStrategy(examples=[...])

    # Use in Env
    env = SomeEnv(..., strategy=strategy)

    # Strategy handles message building
    messages = strategy.build_messages(question)

    # Optional output postprocessing before reward
    processed = strategy.postprocess_output(raw_output)
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

# Message type alias compatible with tinker_cookbook.renderers.Message
# at runtime but simpler for local type checking
Message = dict[str, str]


@runtime_checkable
class PromptStrategy(Protocol):
    """
    First-class prompt/harness object `s` for the (s, θ) optimization.

    This protocol is dataset-agnostic: it does not assume any particular
    answer format (boxed, delimiters, etc.). The reward function R(x, y)
    remains external to the strategy.

    The strategy controls:
    - How to build the message list from a question (few-shot, system prompt, etc.)
    - Optional output postprocessing before reward computation

    Both rollout generation and evaluation use the same strategy instance.
    """

    def build_messages(self, question: str) -> list[Message]:
        """
        Build the full message list for a question.

        Args:
            question: The formatted question string. The question formatting
                     (e.g., adding "Write your answer in \\boxed{} format")
                     is handled by the Env, not the strategy.

        Returns:
            Full message list including any few-shot examples, system prompts, etc.
            Format: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
        """
        ...

    def postprocess_output(self, raw_output: str) -> str:
        """
        Optional output postprocessing before reward computation.

        This can be used for extraction, normalization, etc.
        The reward function R(x, y) receives the postprocessed output.

        Args:
            raw_output: The model's raw generation

        Returns:
            Processed output to pass to reward function
        """
        ...

    @property
    def name(self) -> str:
        """Unique identifier for this strategy (for logging/serialization)."""
        ...


@dataclass
class FewShotStrategy:
    """
    Strategy that prepends few-shot examples to the question.

    This is the most common strategy type, equivalent to the current
    `convo_prefix` behavior.
    """

    fewshot_messages: list[Message] = field(default_factory=list)
    strategy_name: str = "fewshot"
    output_postprocessor: Callable[[str], str] | None = None

    def build_messages(self, question: str) -> list[Message]:
        """Build messages with few-shot prefix + user question."""
        return self.fewshot_messages + [{"role": "user", "content": question}]

    def postprocess_output(self, raw_output: str) -> str:
        """Apply optional postprocessor or return unchanged."""
        if self.output_postprocessor is not None:
            return self.output_postprocessor(raw_output)
        return raw_output

    @property
    def name(self) -> str:
        return self.strategy_name


@dataclass
class SystemPromptStrategy:
    """
    Strategy that uses a system prompt plus optional few-shot examples.
    """

    system_prompt: str
    fewshot_messages: list[Message] = field(default_factory=list)
    strategy_name: str = "system_prompt"
    output_postprocessor: Callable[[str], str] | None = None

    def build_messages(self, question: str) -> list[Message]:
        """Build messages with system prompt + few-shot + user question."""
        messages: list[Message] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.fewshot_messages)
        messages.append({"role": "user", "content": question})
        return messages

    def postprocess_output(self, raw_output: str) -> str:
        if self.output_postprocessor is not None:
            return self.output_postprocessor(raw_output)
        return raw_output

    @property
    def name(self) -> str:
        return self.strategy_name


@dataclass
class NoPromptStrategy:
    """
    Minimal strategy: just the question, no few-shot or system prompt.
    """

    strategy_name: str = "none"
    output_postprocessor: Callable[[str], str] | None = None

    def build_messages(self, question: str) -> list[Message]:
        """Build messages with just the user question."""
        return [{"role": "user", "content": question}]

    def postprocess_output(self, raw_output: str) -> str:
        if self.output_postprocessor is not None:
            return self.output_postprocessor(raw_output)
        return raw_output

    @property
    def name(self) -> str:
        return self.strategy_name


# =============================================================================
# Conversion utilities for backward compatibility
# =============================================================================

def convo_prefix_to_strategy(
    convo_prefix: list[Message] | None,
    name: str = "legacy",
) -> PromptStrategy:
    """
    Convert a convo_prefix (list of messages) to a PromptStrategy.

    This provides backward compatibility with the existing API.
    """
    if convo_prefix is None:
        return NoPromptStrategy(strategy_name=name)
    return FewShotStrategy(fewshot_messages=convo_prefix, strategy_name=name)


def strategy_to_convo_prefix(strategy: PromptStrategy) -> list[Message] | None:
    """
    Extract convo_prefix from a strategy (for backward compatibility).

    Returns None if the strategy doesn't have few-shot messages.
    """
    if isinstance(strategy, FewShotStrategy):
        return strategy.fewshot_messages if strategy.fewshot_messages else None
    if isinstance(strategy, SystemPromptStrategy):
        # Include system message + few-shot
        messages = [{"role": "system", "content": strategy.system_prompt}]
        messages.extend(strategy.fewshot_messages)
        return messages if messages else None
    if isinstance(strategy, NoPromptStrategy):
        return None
    # For unknown strategy types, return None
    return None


# =============================================================================
# Strategy registry for named strategies
# =============================================================================

_STRATEGY_REGISTRY: dict[str, Callable[[], PromptStrategy]] = {}


def register_strategy(name: str, factory: Callable[[], PromptStrategy]) -> None:
    """Register a strategy factory by name."""
    _STRATEGY_REGISTRY[name] = factory


def get_strategy(name: str) -> PromptStrategy:
    """Get a strategy by name."""
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[name]()


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    return list(_STRATEGY_REGISTRY.keys())
