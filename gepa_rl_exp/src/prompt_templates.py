"""
Prompt template definitions for math RL training.

This module provides prompt strategies for the (s, θ) joint optimization.
Each strategy defines how to build messages for a problem and optionally
how to postprocess model outputs.

The strategies here are examples for math tasks, but the PromptStrategy
interface is dataset-agnostic.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

from .prompt_strategy import (
    FewShotStrategy,
    NoPromptStrategy,
    PromptStrategy,
    register_strategy,
)

# Type alias for message format (matches tinker_cookbook.renderers.Message)
Message = dict[str, str]

# Available template names (for backward compatibility)
TemplateName = Literal["baseline", "step_by_step", "concise", "none"]


# =============================================================================
# Few-shot example definitions
# =============================================================================

BASELINE_FEWSHOT: list[Message] = [
    {
        "role": "user",
        "content": "How many r's are in strawberry? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
    },
]

STEP_BY_STEP_FEWSHOT: list[Message] = [
    {
        "role": "user",
        "content": "If a store sells apples for $2 each and oranges for $3 each, how much would 4 apples and 3 oranges cost? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": """Let me solve this step by step.

Step 1: Calculate the cost of apples
- Price per apple: $2
- Number of apples: 4
- Cost of apples: $2 × 4 = $8

Step 2: Calculate the cost of oranges
- Price per orange: $3
- Number of oranges: 3
- Cost of oranges: $3 × 3 = $9

Step 3: Calculate the total cost
- Total = Cost of apples + Cost of oranges
- Total = $8 + $9 = $17

\\boxed{17}""",
    },
]

CONCISE_FEWSHOT: list[Message] = [
    {
        "role": "user",
        "content": "What is 15% of 80? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": "15% of 80 = 0.15 × 80 = 12. \\boxed{12}",
    },
]


# =============================================================================
# Strategy instances
# =============================================================================


def create_baseline_strategy() -> PromptStrategy:
    """Baseline strategy with simple enumeration example."""
    return FewShotStrategy(
        fewshot_messages=BASELINE_FEWSHOT,
        strategy_name="baseline",
    )


def create_step_by_step_strategy() -> PromptStrategy:
    """Verbose step-by-step reasoning strategy."""
    return FewShotStrategy(
        fewshot_messages=STEP_BY_STEP_FEWSHOT,
        strategy_name="step_by_step",
    )


def create_concise_strategy() -> PromptStrategy:
    """Minimal reasoning strategy for quick answers."""
    return FewShotStrategy(
        fewshot_messages=CONCISE_FEWSHOT,
        strategy_name="concise",
    )


def create_no_prompt_strategy() -> PromptStrategy:
    """No few-shot examples, just the question."""
    return NoPromptStrategy(strategy_name="none")


# Register strategies
register_strategy("baseline", create_baseline_strategy)
register_strategy("step_by_step", create_step_by_step_strategy)
register_strategy("concise", create_concise_strategy)
register_strategy("none", create_no_prompt_strategy)


# =============================================================================
# Backward compatibility API
# =============================================================================


def get_template(name: TemplateName | str) -> list[Message]:
    """
    Get a prompt template by name.

    DEPRECATED: Use get_strategy() instead for the full PromptStrategy object.

    Args:
        name: Template name - one of "baseline", "step_by_step", "concise", "none"

    Returns:
        List of messages to use as convo_prefix (few-shot examples)
    """
    templates: dict[str, list[Message]] = {
        "baseline": BASELINE_FEWSHOT,
        "step_by_step": STEP_BY_STEP_FEWSHOT,
        "concise": CONCISE_FEWSHOT,
        "none": [],
    }
    if name not in templates:
        raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")
    return templates[name]


def get_strategy_by_name(name: TemplateName | str) -> PromptStrategy:
    """
    Get a PromptStrategy by name.

    Args:
        name: Strategy name - one of "baseline", "step_by_step", "concise", "none"

    Returns:
        PromptStrategy instance
    """
    from .prompt_strategy import get_strategy

    return get_strategy(name)


# Legacy aliases
BASELINE_TEMPLATE = BASELINE_FEWSHOT
STEP_BY_STEP_TEMPLATE = STEP_BY_STEP_FEWSHOT
CONCISE_TEMPLATE = CONCISE_FEWSHOT
