"""
Hint generation module with templates and safeguards.

This module provides:
1. A dataset-agnostic prompt template for hint generation
2. Safeguards to prevent answer leakage in hints
3. Utilities for sanitizing and validating hints
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .teacher_client import TeacherClient


# =============================================================================
# Hint Generation Prompt Template
# =============================================================================

HINT_GENERATION_PROMPT_TEMPLATE = """You are a helpful tutor reviewing a student's work on a problem.

## Problem:
{question}

## Student's Attempt:
{student_response}

## Your Task:
Provide a helpful hint that guides the student toward the correct solution WITHOUT revealing the answer directly.

CRITICAL RULES - You MUST follow these:
1. DO NOT state the final answer or any intermediate numerical results that would give away the solution
2. DO NOT copy or paraphrase the reference solution
3. Point out WHERE the error might be (e.g., "Check your calculation in the second step")
4. Suggest a STRATEGY or METHOD to find and fix the error
5. Keep the hint concise (under {max_chars} characters)
6. Focus on guiding the student's thinking, not solving for them

Examples of GOOD hints:
- "Check your arithmetic when you added those two numbers"
- "Consider whether you applied the distributive property correctly"
- "Review the units in your calculation"
- "Think about what the problem is actually asking for"

Examples of BAD hints (DO NOT do these):
- "The answer is 42" (reveals answer)
- "You should get 15 in step 2" (reveals intermediate result)
- "The correct calculation is 5+3=8" (solves for student)

Hint:"""


def build_hint_prompt(
    question: str,
    student_response: str,
    ground_truth: str,  # For context, but NOT included in prompt
    max_chars: int = 500,
) -> str:
    """
    Build the prompt for hint generation.

    Note: ground_truth is NOT included in the prompt to prevent the teacher
    from accidentally copying it. The teacher must infer what's wrong from
    comparing the student's work to general problem-solving principles.
    """
    return HINT_GENERATION_PROMPT_TEMPLATE.format(
        question=question,
        student_response=student_response,
        max_chars=max_chars,
    )


# =============================================================================
# Hint Safeguards
# =============================================================================


@dataclass
class HintRequest:
    """Request for generating a hint."""

    question: str
    student_response: str
    ground_truth: str  # For safeguard checking only


@dataclass
class HintResult:
    """Result of hint generation with safeguard information."""

    hint: str  # Final sanitized hint
    raw_hint: str  # Original hint before safeguards
    safeguard_applied: bool
    safeguard_reason: str | None


class HintSafeguards:
    """Validates and sanitizes hints to prevent answer leakage."""

    @staticmethod
    def extract_numbers(text: str) -> list[str]:
        """Extract all number-like tokens from text."""
        # Match integers, decimals, fractions, and scientific notation
        patterns = [
            r"-?\d+\.?\d*",  # Integers and decimals
            r"-?\d+/\d+",  # Fractions
            r"-?\d+e[+-]?\d+",  # Scientific notation
        ]
        numbers = []
        for pattern in patterns:
            numbers.extend(re.findall(pattern, text, re.IGNORECASE))
        return [n.strip() for n in numbers if n.strip()]

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison (lowercase, strip whitespace)."""
        return answer.lower().strip()

    @staticmethod
    def check_answer_leak(
        hint: str,
        ground_truth: str,
        strictness: Literal["strict", "moderate", "lenient"],
    ) -> tuple[bool, str | None]:
        """
        Check if hint leaks the answer.

        Args:
            hint: The generated hint
            ground_truth: The correct answer
            strictness: How strict to be about potential leaks
                - strict: Block ANY numeric token from ground_truth
                - moderate: Block exact match or >3 consecutive digits
                - lenient: Block only exact answer string

        Returns:
            (is_safe, reason_if_unsafe)
        """
        hint_lower = HintSafeguards.normalize_answer(hint)
        gt_lower = HintSafeguards.normalize_answer(ground_truth)

        # Always check for exact match
        if gt_lower in hint_lower:
            return False, f"Exact answer '{ground_truth}' found in hint"

        if strictness == "lenient":
            return True, None

        # Extract numbers from both
        hint_numbers = set(HintSafeguards.extract_numbers(hint))
        gt_numbers = set(HintSafeguards.extract_numbers(ground_truth))

        if strictness == "strict":
            # Block if ANY number from ground_truth appears in hint
            overlap = hint_numbers & gt_numbers
            if overlap:
                return False, f"Numbers from answer found in hint: {overlap}"

        elif strictness == "moderate":
            # Block if exact number match or long numeric sequences
            overlap = hint_numbers & gt_numbers
            if overlap:
                # Check if it's a "significant" number (not just 0, 1, 2, etc.)
                significant_overlap = {n for n in overlap if len(n) > 2 or int(float(n)) > 10}
                if significant_overlap:
                    return False, f"Significant numbers from answer in hint: {significant_overlap}"

            # Also check for long consecutive digit sequences
            gt_digits = re.sub(r"[^\d]", "", ground_truth)
            hint_digits = re.sub(r"[^\d]", "", hint)
            if len(gt_digits) >= 3:
                for i in range(len(gt_digits) - 2):
                    seq = gt_digits[i : i + 3]
                    if seq in hint_digits:
                        return False, f"Digit sequence '{seq}' from answer found in hint"

        return True, None

    @staticmethod
    def truncate_hint(hint: str, max_chars: int) -> str:
        """Truncate hint to max length, preserving sentence boundaries."""
        if len(hint) <= max_chars:
            return hint

        # Try to cut at sentence boundary
        truncated = hint[:max_chars]
        last_period = truncated.rfind(".")
        last_question = truncated.rfind("?")
        last_exclaim = truncated.rfind("!")

        best_cut = max(last_period, last_question, last_exclaim)
        if best_cut > max_chars * 0.5:  # Only use if we keep at least half
            return truncated[: best_cut + 1]

        # Otherwise just truncate and add ellipsis
        return truncated[: max_chars - 3] + "..."

    @staticmethod
    def sanitize_hint(
        hint: str,
        ground_truth: str,
        strictness: Literal["strict", "moderate", "lenient"],
        max_chars: int,
    ) -> HintResult:
        """
        Apply all safeguards and return result.

        If the hint fails safeguards, returns a generic fallback hint.
        """
        raw_hint = hint

        # First truncate
        hint = HintSafeguards.truncate_hint(hint, max_chars)

        # Check for answer leakage
        is_safe, reason = HintSafeguards.check_answer_leak(hint, ground_truth, strictness)

        if not is_safe:
            # Return a generic fallback hint
            return HintResult(
                hint="Review your approach and check each step carefully.",
                raw_hint=raw_hint,
                safeguard_applied=True,
                safeguard_reason=reason,
            )

        return HintResult(
            hint=hint,
            raw_hint=raw_hint,
            safeguard_applied=False,
            safeguard_reason=None,
        )


# =============================================================================
# High-Level API
# =============================================================================


async def generate_hint(
    client: "TeacherClient",
    request: HintRequest,
    max_chars: int = 500,
    safeguard_strictness: Literal["strict", "moderate", "lenient"] = "moderate",
) -> HintResult:
    """
    Generate a hint using the teacher client with safeguards applied.

    Args:
        client: Teacher client (OpenAI, Anthropic, or Dummy)
        request: HintRequest with question, student_response, ground_truth
        max_chars: Maximum hint length
        safeguard_strictness: How strict to be about answer leakage

    Returns:
        HintResult with sanitized hint and safeguard info
    """
    # Generate raw hint from teacher
    raw_hint = await client.generate_hint(
        question=request.question,
        student_response=request.student_response,
        ground_truth=request.ground_truth,
    )

    # Apply safeguards
    result = HintSafeguards.sanitize_hint(
        hint=raw_hint,
        ground_truth=request.ground_truth,
        strictness=safeguard_strictness,
        max_chars=max_chars,
    )

    return result
