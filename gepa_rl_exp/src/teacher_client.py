"""
External API clients for teacher models (OpenAI/Anthropic).

These clients are used to generate hints during expert iteration.
The teacher model receives (question, student_response, ground_truth)
and returns an actionable hint that doesn't reveal the answer.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass
class TeacherConfig:
    """Configuration for teacher model."""

    backend: Literal["openai", "anthropic"]
    model: str  # e.g., "gpt-4o", "claude-3-5-sonnet-20241022"
    max_hint_chars: int = 500
    temperature: float = 0.7
    api_key_env_var: str | None = None  # Falls back to standard env vars


@runtime_checkable
class TeacherClient(Protocol):
    """Protocol for external teacher API."""

    async def generate_hint(
        self,
        question: str,
        student_response: str,
        ground_truth: str,
    ) -> str:
        """
        Generate a hint based on student's attempt.

        Args:
            question: The problem/question text
            student_response: Student model's response (y0)
            ground_truth: The correct answer (for teacher context only)

        Returns:
            A hint string that guides the student without revealing the answer
        """
        ...


class OpenAITeacherClient:
    """OpenAI API implementation for teacher hints."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        self._client: "openai.AsyncOpenAI | None" = None

    def _get_client(self) -> "openai.AsyncOpenAI":
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI teacher. "
                    "Install with: pip install openai"
                )

            api_key = None
            if self.config.api_key_env_var:
                api_key = os.environ.get(self.config.api_key_env_var)
            # Falls back to OPENAI_API_KEY automatically

            self._client = openai.AsyncOpenAI(api_key=api_key)
        return self._client

    async def generate_hint(
        self,
        question: str,
        student_response: str,
        ground_truth: str,
    ) -> str:
        from .hint_generation import build_hint_prompt

        client = self._get_client()
        prompt = build_hint_prompt(
            question=question,
            student_response=student_response,
            ground_truth=ground_truth,
            max_chars=self.config.max_hint_chars,
        )

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_hint_chars,
            temperature=self.config.temperature,
        )

        return response.choices[0].message.content or ""


class AnthropicTeacherClient:
    """Anthropic API implementation for teacher hints."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        self._client: "anthropic.AsyncAnthropic | None" = None

    def _get_client(self) -> "anthropic.AsyncAnthropic":
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic teacher. "
                    "Install with: pip install anthropic"
                )

            api_key = None
            if self.config.api_key_env_var:
                api_key = os.environ.get(self.config.api_key_env_var)
            # Falls back to ANTHROPIC_API_KEY automatically

            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._client

    async def generate_hint(
        self,
        question: str,
        student_response: str,
        ground_truth: str,
    ) -> str:
        from .hint_generation import build_hint_prompt

        client = self._get_client()
        prompt = build_hint_prompt(
            question=question,
            student_response=student_response,
            ground_truth=ground_truth,
            max_chars=self.config.max_hint_chars,
        )

        response = await client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_hint_chars,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from response
        if response.content and len(response.content) > 0:
            return response.content[0].text
        return ""


class DummyTeacherClient:
    """
    Dummy teacher client for testing.

    Always returns a fixed hint, useful for unit tests and debugging.
    """

    def __init__(self, fixed_hint: str = "Check your work carefully."):
        self.fixed_hint = fixed_hint

    async def generate_hint(
        self,
        question: str,
        student_response: str,
        ground_truth: str,
    ) -> str:
        return self.fixed_hint


def create_teacher_client(config: TeacherConfig) -> TeacherClient:
    """Factory function to create appropriate teacher client."""
    if config.backend == "openai":
        return OpenAITeacherClient(config)
    elif config.backend == "anthropic":
        return AnthropicTeacherClient(config)
    else:
        raise ValueError(f"Unknown teacher backend: {config.backend}")
