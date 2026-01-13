"""
PromptStrategy implementations for hint injection during expert iteration.

This module provides:
- HintStrategy: Injects hints into prompts (used ONLY during expert refinement)
- CleanStrategy: Ensures NO hint in prompts (used for final distillation)

The key invariant is that the student model never sees hints during distillation,
ensuring it learns to produce good answers without external guidance.
"""

from dataclasses import dataclass

from .prompt_strategy import Message, PromptStrategy


@dataclass
class HintStrategy:
    """
    PromptStrategy that includes a teacher hint in the prompt.

    Used ONLY during expert refinement sampling, NOT during final distillation.
    Implements the PromptStrategy protocol from prompt_strategy.py.
    """

    base_strategy: PromptStrategy
    hint: str
    strategy_name: str = "hint_augmented"
    # How to inject the hint into the conversation
    injection_mode: str = "assistant_turn"  # "assistant_turn" or "inline"

    def build_messages(self, question: str) -> list[Message]:
        """
        Build messages with hint context.

        The hint is injected as an assistant turn after the question,
        simulating a tutor providing guidance before the student's response.
        """
        base_messages = self.base_strategy.build_messages(question)
        return self._inject_hint(base_messages, self.hint)

    def _inject_hint(self, messages: list[Message], hint: str) -> list[Message]:
        """
        Inject hint into message list.

        Strategy: Add hint as an assistant turn after the user's question.
        This simulates the teacher giving guidance after seeing the problem
        but before the student produces their final answer.
        """
        if not hint:
            return messages

        result = list(messages)  # Copy to avoid mutation

        if self.injection_mode == "assistant_turn":
            # Add hint as a separate assistant turn
            # Find the last user message and insert hint after it
            hint_message: Message = {
                "role": "assistant",
                "content": f"Hint: {hint}",
            }
            # Insert before any trailing assistant message, or append
            # Typically messages end with user question, so we append
            result.append(hint_message)

        elif self.injection_mode == "inline":
            # Append hint to the last user message
            if result and result[-1]["role"] == "user":
                result[-1] = {
                    "role": "user",
                    "content": result[-1]["content"] + f"\n\n[Hint: {hint}]",
                }
            else:
                # Fallback to assistant turn
                result.append({"role": "assistant", "content": f"Hint: {hint}"})

        return result

    def postprocess_output(self, raw_output: str) -> str:
        """Pass through to base strategy's postprocessing."""
        return self.base_strategy.postprocess_output(raw_output)

    @property
    def name(self) -> str:
        return self.strategy_name


@dataclass
class CleanStrategy:
    """
    Wrapper that ensures NO hint is present in the prompt.

    Used for final distillation to guarantee the student never sees hints
    at inference time. This is critical for the expert iteration approach:
    the student learns to produce y* without the hint that was used to
    generate it.
    """

    base_strategy: PromptStrategy
    strategy_name: str | None = None

    def build_messages(self, question: str) -> list[Message]:
        """Build messages WITHOUT any hint - just the base strategy."""
        return self.base_strategy.build_messages(question)

    def postprocess_output(self, raw_output: str) -> str:
        """Pass through to base strategy's postprocessing."""
        return self.base_strategy.postprocess_output(raw_output)

    @property
    def name(self) -> str:
        if self.strategy_name:
            return self.strategy_name
        return f"{self.base_strategy.name}_clean"


def create_hint_strategy(
    base_strategy: PromptStrategy,
    hint: str,
    injection_mode: str = "assistant_turn",
) -> HintStrategy:
    """
    Factory function to create a HintStrategy.

    Args:
        base_strategy: The underlying strategy without hints
        hint: The hint text to inject
        injection_mode: How to inject the hint ("assistant_turn" or "inline")

    Returns:
        HintStrategy instance
    """
    return HintStrategy(
        base_strategy=base_strategy,
        hint=hint,
        strategy_name=f"{base_strategy.name}_with_hint",
        injection_mode=injection_mode,
    )


def create_clean_strategy(base_strategy: PromptStrategy) -> CleanStrategy:
    """
    Factory function to create a CleanStrategy.

    Args:
        base_strategy: The underlying strategy

    Returns:
        CleanStrategy instance that guarantees no hint
    """
    return CleanStrategy(base_strategy=base_strategy)
