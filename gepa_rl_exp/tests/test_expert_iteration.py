"""
Unit and integration tests for expert iteration.

Tests cover:
1. Hint safeguards (answer leakage detection)
2. HintStrategy includes hint in messages
3. CleanStrategy excludes hint in distillation
4. Expert iteration flow with dummy teacher
"""

import asyncio
import pytest

# Import modules under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hint_generation import HintRequest, HintResult, HintSafeguards
from src.hint_strategy import CleanStrategy, HintStrategy, create_clean_strategy, create_hint_strategy
from src.prompt_strategy import FewShotStrategy, NoPromptStrategy
from src.teacher_client import DummyTeacherClient


class TestHintSafeguards:
    """Test hint safeguard functions."""

    def test_exact_answer_in_hint_is_blocked(self):
        """Verify safeguards catch exact answer in hint."""
        hint = "The answer is 42, you should get that."
        ground_truth = "42"

        is_safe, reason = HintSafeguards.check_answer_leak(hint, ground_truth, "lenient")
        assert not is_safe
        assert "42" in reason

    def test_partial_number_match_strict(self):
        """Strict mode blocks any numeric token from ground_truth."""
        hint = "Check the step where you got 15"
        ground_truth = "15"

        is_safe, _ = HintSafeguards.check_answer_leak(hint, ground_truth, "strict")
        assert not is_safe

    def test_safe_hint_passes(self):
        """Safe hint without answer passes all modes."""
        hint = "Check your arithmetic in the second step."
        ground_truth = "42"

        for strictness in ["strict", "moderate", "lenient"]:
            is_safe, _ = HintSafeguards.check_answer_leak(hint, ground_truth, strictness)
            assert is_safe, f"Safe hint should pass {strictness} mode"

    def test_moderate_blocks_exact_match(self):
        """Moderate mode blocks exact answer match even for small numbers."""
        # The exact answer string check happens BEFORE numeric checks
        hint = "Check your work on step 2"
        ground_truth = "2"

        is_safe, reason = HintSafeguards.check_answer_leak(hint, ground_truth, "moderate")
        # Even though "2" is small, exact string match is blocked
        assert not is_safe
        assert "Exact answer" in reason

    def test_sanitize_hint_fallback(self):
        """Sanitize returns fallback when hint is unsafe."""
        hint = "The answer is definitely 42"
        ground_truth = "42"

        result = HintSafeguards.sanitize_hint(hint, ground_truth, "lenient", max_chars=500)
        assert result.safeguard_applied
        assert "42" not in result.hint  # Fallback hint
        assert result.raw_hint == hint

    def test_truncate_hint(self):
        """Test hint truncation."""
        long_hint = "A" * 1000
        truncated = HintSafeguards.truncate_hint(long_hint, max_chars=100)
        assert len(truncated) <= 100

    def test_extract_numbers(self):
        """Test number extraction from text."""
        text = "Step 1: 42 + 3.14 = 45.14"
        numbers = HintSafeguards.extract_numbers(text)
        assert "42" in numbers
        assert "3.14" in numbers
        assert "45.14" in numbers


class TestHintStrategy:
    """Test HintStrategy includes hint in messages."""

    def test_hint_included_in_messages_assistant_turn(self):
        """HintStrategy adds hint as assistant turn."""
        base = FewShotStrategy(fewshot_messages=[], strategy_name="base")
        hint_strategy = create_hint_strategy(base, hint="Check your addition", injection_mode="assistant_turn")

        messages = hint_strategy.build_messages("What is 2+2?")

        # Should have user question + assistant hint
        assert len(messages) >= 2
        # Find the hint message
        hint_messages = [m for m in messages if "Hint:" in m.get("content", "")]
        assert len(hint_messages) == 1
        assert "Check your addition" in hint_messages[0]["content"]

    def test_hint_included_inline(self):
        """HintStrategy can add hint inline to user message."""
        base = NoPromptStrategy(strategy_name="none")
        hint_strategy = create_hint_strategy(base, hint="Check your work", injection_mode="inline")

        messages = hint_strategy.build_messages("What is 2+2?")

        # Hint should be appended to user message
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert "[Hint: Check your work]" in user_messages[0]["content"]

    def test_empty_hint_no_change(self):
        """Empty hint doesn't modify messages."""
        base = NoPromptStrategy(strategy_name="none")
        hint_strategy = HintStrategy(base_strategy=base, hint="")

        messages = hint_strategy.build_messages("What is 2+2?")

        # Only the user question
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


class TestCleanStrategy:
    """Test CleanStrategy excludes hint."""

    def test_clean_strategy_no_hint(self):
        """CleanStrategy builds messages without any hint."""
        base = FewShotStrategy(
            fewshot_messages=[
                {"role": "user", "content": "Example Q"},
                {"role": "assistant", "content": "Example A"},
            ],
            strategy_name="fewshot",
        )
        clean = create_clean_strategy(base)

        messages = clean.build_messages("What is 2+2?")

        # Should match base strategy exactly
        base_messages = base.build_messages("What is 2+2?")
        assert messages == base_messages

        # No hint anywhere
        for m in messages:
            assert "hint" not in m.get("content", "").lower()

    def test_clean_strategy_name(self):
        """CleanStrategy has appropriate name."""
        base = NoPromptStrategy(strategy_name="test")
        clean = CleanStrategy(base_strategy=base)

        assert "clean" in clean.name


class TestDummyTeacherClient:
    """Test DummyTeacherClient for testing."""

    @pytest.mark.asyncio
    async def test_dummy_returns_fixed_hint(self):
        """DummyTeacherClient returns the fixed hint."""
        client = DummyTeacherClient(fixed_hint="Custom test hint")

        hint = await client.generate_hint(
            question="What is 2+2?",
            student_response="5",
            ground_truth="4",
        )

        assert hint == "Custom test hint"

    @pytest.mark.asyncio
    async def test_dummy_default_hint(self):
        """DummyTeacherClient has default hint."""
        client = DummyTeacherClient()

        hint = await client.generate_hint(
            question="What is 2+2?",
            student_response="5",
            ground_truth="4",
        )

        assert len(hint) > 0  # Has some default hint


class TestExpertIterationInvariants:
    """
    Test critical invariants of expert iteration.

    These tests verify that:
    1. Distillation uses CLEAN strategy (no hint)
    2. Expert refinement uses hint
    """

    def test_distillation_datum_uses_clean_strategy(self):
        """CRITICAL: Verify distillation datums don't include hint."""
        # This is the core invariant: student never sees hint at inference
        base = FewShotStrategy(
            fewshot_messages=[
                {"role": "user", "content": "What is 1+1?"},
                {"role": "assistant", "content": "2"},
            ],
            strategy_name="test",
        )

        # Create a hint strategy (used during expert refinement)
        hint_strategy = create_hint_strategy(base, hint="Check arithmetic")

        # Create a clean strategy (used for distillation)
        clean_strategy = create_clean_strategy(base)

        # Verify hint_strategy includes hint
        hint_messages = hint_strategy.build_messages("What is 2+2?")
        hint_present = any("Hint:" in m.get("content", "") for m in hint_messages)
        assert hint_present, "HintStrategy should include hint"

        # Verify clean_strategy excludes hint
        clean_messages = clean_strategy.build_messages("What is 2+2?")
        hint_in_clean = any("hint" in m.get("content", "").lower() for m in clean_messages)
        assert not hint_in_clean, "CleanStrategy should NOT include hint"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
