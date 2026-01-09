"""
Prompt template definitions for math RL training.

Templates are defined as convo_prefix message lists that prime the model
for different reasoning styles. The question_suffix (boxed format instruction)
is handled by MathEnv and should not be duplicated here.
"""

from typing import Literal

# Type alias for message format (matches tinker_cookbook.renderers.Message)
Message = dict[str, str]

# Available template names
TemplateName = Literal["baseline", "step_by_step", "concise"]


def get_template(name: TemplateName | str) -> list[Message]:
    """Get a prompt template by name.

    Args:
        name: Template name - one of "baseline", "step_by_step", "concise"

    Returns:
        List of messages to use as convo_prefix (few-shot examples)

    Raises:
        ValueError: If template name is unknown
    """
    templates: dict[str, list[Message]] = {
        "baseline": BASELINE_TEMPLATE,
        "step_by_step": STEP_BY_STEP_TEMPLATE,
        "concise": CONCISE_TEMPLATE,
    }
    if name not in templates:
        raise ValueError(f"Unknown template: {name}. Available: {list(templates.keys())}")
    return templates[name]


# =============================================================================
# Template 1: Baseline
# =============================================================================
# Same as MathEnv.standard_fewshot_prefix() - simple problem with enumeration
BASELINE_TEMPLATE: list[Message] = [
    {
        "role": "user",
        "content": "How many r's are in strawberry? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
    },
]


# =============================================================================
# Template 2: Step-by-Step (verbose reasoning)
# =============================================================================
# Emphasizes explicit step-by-step mathematical reasoning with clear labels
STEP_BY_STEP_TEMPLATE: list[Message] = [
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


# =============================================================================
# Template 3: Concise (minimal reasoning)
# =============================================================================
# Direct computation with brief explanation - for when verbosity hurts
CONCISE_TEMPLATE: list[Message] = [
    {
        "role": "user",
        "content": "What is 15% of 80? Write your answer in \\boxed{} format.",
    },
    {
        "role": "assistant",
        "content": "15% of 80 = 0.15 × 80 = 12. \\boxed{12}",
    },
]
