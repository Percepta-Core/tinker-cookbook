"""
LLM-based reflection for GEPA prompt optimization.

The reflection LM analyzes execution traces to:
1. Identify patterns in failures
2. Diagnose root causes
3. Propose concrete prompt improvements

This is the core innovation of GEPA (arxiv:2507.19457) - using natural
language reflection on traces rather than blind search.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from .trace import ExecutionTrace, format_traces_for_reflection


# System prompt for the reflection LM
REFLECTION_SYSTEM_PROMPT = """You are a prompt optimization expert analyzing execution traces from an LLM system.

Your task is to:
1. Analyze the provided execution traces (both successes and failures)
2. Identify patterns in the failures - what goes wrong and why
3. Propose a concrete improvement to the system prompt or few-shot examples

Focus on actionable changes that address the observed failure patterns.

IMPORTANT GUIDELINES:
- Be specific about what to change in the prompt
- Target the most common or impactful failure patterns
- Keep the proposed prompt concise but effective
- Do not assume any specific task format - the prompt should be task-agnostic
- Base your proposal on the actual failures observed, not hypothetical issues

Your response must be valid JSON with this structure:
{
    "failure_analysis": "Brief analysis of what's going wrong (2-3 sentences)",
    "failure_patterns": ["pattern1", "pattern2", ...],
    "proposed_prompt": "The complete new system prompt or instruction to use",
    "reasoning": "Why this change should help (1-2 sentences)"
}"""


USER_PROMPT_TEMPLATE = """Here is the current prompt being used:

<current_prompt>
{current_prompt}
</current_prompt>

Here are the execution traces from recent rollouts:

{traces}

Based on these traces, propose an improved prompt. Remember to return valid JSON."""


@dataclass
class PromptProposal:
    """
    A proposed prompt improvement from the reflection LM.

    Contains both the new prompt text and metadata about why
    the change was proposed.
    """

    new_prompt: str  # The proposed system prompt / instruction
    reasoning: str  # Why this change should help
    failure_analysis: str  # Analysis of what was going wrong
    failure_patterns: list[str]  # Specific patterns identified

    def to_dict(self) -> dict:
        return {
            "new_prompt": self.new_prompt,
            "reasoning": self.reasoning,
            "failure_analysis": self.failure_analysis,
            "failure_patterns": self.failure_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PromptProposal":
        return cls(
            new_prompt=data["new_prompt"],
            reasoning=data.get("reasoning", ""),
            failure_analysis=data.get("failure_analysis", ""),
            failure_patterns=data.get("failure_patterns", []),
        )


def parse_reflection_response(response: str) -> PromptProposal:
    """
    Parse the JSON response from the reflection LM.

    Handles common formatting issues like markdown code blocks.
    """
    # Strip markdown code blocks if present
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse reflection response as JSON: {e}")
        else:
            raise ValueError(f"Could not parse reflection response as JSON: {e}")

    # Validate required field
    if "proposed_prompt" not in data:
        raise ValueError("Reflection response missing 'proposed_prompt' field")

    return PromptProposal(
        new_prompt=data["proposed_prompt"],
        reasoning=data.get("reasoning", ""),
        failure_analysis=data.get("failure_analysis", ""),
        failure_patterns=data.get("failure_patterns", []),
    )


class ReflectionClient:
    """
    Client for calling the reflection LM.

    Supports OpenAI and Anthropic APIs. Uses a similar pattern to
    TeacherClient but with a different interface for reflection.
    """

    def __init__(
        self,
        backend: Literal["openai", "anthropic"],
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI reflection. "
                    "Install with: pip install openai"
                )
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    def _get_anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Anthropic reflection. "
                    "Install with: pip install anthropic"
                )
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def reflect(
        self,
        traces: list[ExecutionTrace],
        current_prompt: str,
    ) -> PromptProposal:
        """
        Call the reflection LM to analyze traces and propose an improvement.

        Args:
            traces: List of execution traces to analyze
            current_prompt: The current prompt being used

        Returns:
            PromptProposal with the suggested improvement
        """
        # Format traces for the LM
        traces_text = format_traces_for_reflection(traces)

        # Build user message
        user_message = USER_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt,
            traces=traces_text,
        )

        if self.backend == "openai":
            return await self._reflect_openai(user_message)
        elif self.backend == "anthropic":
            return await self._reflect_anthropic(user_message)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    async def _reflect_openai(self, user_message: str) -> PromptProposal:
        client = self._get_openai_client()

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        content = response.choices[0].message.content or ""
        return parse_reflection_response(content)

    async def _reflect_anthropic(self, user_message: str) -> PromptProposal:
        client = self._get_anthropic_client()

        response = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=REFLECTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        if response.content and len(response.content) > 0:
            content = response.content[0].text
        else:
            content = ""

        return parse_reflection_response(content)


class DummyReflectionClient:
    """
    Dummy reflection client for testing.

    Returns a fixed proposal, useful for unit tests and debugging.
    """

    def __init__(self, fixed_prompt: str = "Think step by step before answering."):
        self.fixed_prompt = fixed_prompt
        self.call_count = 0

    async def reflect(
        self,
        traces: list[ExecutionTrace],
        current_prompt: str,
    ) -> PromptProposal:
        self.call_count += 1
        return PromptProposal(
            new_prompt=self.fixed_prompt,
            reasoning="Test proposal",
            failure_analysis="Test analysis",
            failure_patterns=["test_pattern"],
        )


async def reflect_on_traces(
    traces: list[ExecutionTrace],
    current_prompt: str,
    client: ReflectionClient | DummyReflectionClient,
) -> PromptProposal:
    """
    Convenience function to reflect on traces using a client.

    This is the main entry point for GEPA reflection.

    Args:
        traces: List of execution traces to analyze
        current_prompt: The current prompt being used
        client: The reflection client to use

    Returns:
        PromptProposal with the suggested improvement
    """
    return await client.reflect(traces, current_prompt)
