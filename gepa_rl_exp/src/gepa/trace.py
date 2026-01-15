"""
Execution trace capture and serialization for GEPA.

An ExecutionTrace captures a complete rollout trajectory including:
- The input problem
- Messages sent to the model
- Model output
- Reward from verifier R(x,y)
- Any feedback from the environment

Traces are serialized to text for the reflection LM to analyze.
"""

from dataclasses import dataclass, field
from typing import Any
import json

# Message type alias (matches tinker_cookbook.renderers.Message)
Message = dict[str, str]


@dataclass
class ExecutionTrace:
    """
    Complete record of a single rollout for GEPA analysis.

    This is the core data structure that the reflection LM analyzes
    to diagnose failures and propose prompt improvements.
    """

    # Problem identification
    problem_id: str

    # Input
    input_text: str  # The question/problem as presented to model

    # Prompt context
    system_prompt: str | None = None  # System prompt if used
    few_shot_messages: list[Message] = field(default_factory=list)  # Few-shot examples

    # Model interaction
    messages: list[Message] = field(default_factory=list)  # Full message list sent
    output: str = ""  # Model's raw output

    # Outcome
    reward: float = 0.0  # R(x,y) from verifier
    correct: bool = False  # Binary correctness

    # Environment feedback (dataset-agnostic)
    env_feedback: str | None = None  # Error messages, partial credit info, etc.
    expected_answer: str | None = None  # Ground truth (for reflection only, not leaked to student)

    # Metadata
    metrics: dict[str, Any] = field(default_factory=dict)  # Additional metrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "problem_id": self.problem_id,
            "input_text": self.input_text,
            "system_prompt": self.system_prompt,
            "few_shot_messages": self.few_shot_messages,
            "messages": self.messages,
            "output": self.output,
            "reward": self.reward,
            "correct": self.correct,
            "env_feedback": self.env_feedback,
            "expected_answer": self.expected_answer,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionTrace":
        """Deserialize from dictionary."""
        return cls(
            problem_id=data["problem_id"],
            input_text=data["input_text"],
            system_prompt=data.get("system_prompt"),
            few_shot_messages=data.get("few_shot_messages", []),
            messages=data.get("messages", []),
            output=data.get("output", ""),
            reward=data.get("reward", 0.0),
            correct=data.get("correct", False),
            env_feedback=data.get("env_feedback"),
            expected_answer=data.get("expected_answer"),
            metrics=data.get("metrics", {}),
        )


def format_single_trace(trace: ExecutionTrace, include_answer: bool = True) -> str:
    """
    Format a single trace as human-readable text for reflection.

    Args:
        trace: The execution trace to format
        include_answer: Whether to include expected answer (for reflection LM)

    Returns:
        Formatted text representation of the trace
    """
    lines = []

    # Header with outcome
    status = "CORRECT" if trace.correct else "INCORRECT"
    lines.append(f"=== Trace [{trace.problem_id}] - {status} (reward={trace.reward:.2f}) ===")
    lines.append("")

    # Problem
    lines.append("PROBLEM:")
    lines.append(trace.input_text)
    lines.append("")

    # Prompt context (if any)
    if trace.system_prompt:
        lines.append("SYSTEM PROMPT:")
        lines.append(trace.system_prompt)
        lines.append("")

    if trace.few_shot_messages:
        lines.append("FEW-SHOT EXAMPLES:")
        for msg in trace.few_shot_messages:
            lines.append(f"  [{msg['role']}]: {msg['content'][:200]}...")
        lines.append("")

    # Model output
    lines.append("MODEL OUTPUT:")
    lines.append(trace.output)
    lines.append("")

    # Expected answer (for reflection)
    if include_answer and trace.expected_answer:
        lines.append("EXPECTED ANSWER:")
        lines.append(trace.expected_answer)
        lines.append("")

    # Environment feedback
    if trace.env_feedback:
        lines.append("ENVIRONMENT FEEDBACK:")
        lines.append(trace.env_feedback)
        lines.append("")

    return "\n".join(lines)


def format_traces_for_reflection(
    traces: list[ExecutionTrace],
    max_traces: int = 20,
    include_answers: bool = True,
) -> str:
    """
    Format a batch of traces for the reflection LM.

    Organizes traces by outcome (failures first) and provides
    summary statistics to help the reflection LM identify patterns.

    Args:
        traces: List of execution traces
        max_traces: Maximum traces to include (to limit context length)
        include_answers: Whether to include expected answers

    Returns:
        Formatted text for reflection LM input
    """
    if not traces:
        return "No traces available."

    # Compute statistics
    n_total = len(traces)
    n_correct = sum(1 for t in traces if t.correct)
    n_incorrect = n_total - n_correct
    avg_reward = sum(t.reward for t in traces) / n_total if traces else 0.0

    # Separate by outcome
    failures = [t for t in traces if not t.correct]
    successes = [t for t in traces if t.correct]

    lines = []

    # Summary header
    lines.append("=" * 60)
    lines.append("EXECUTION TRACE SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Total traces: {n_total}")
    lines.append(f"Correct: {n_correct} ({100*n_correct/n_total:.1f}%)")
    lines.append(f"Incorrect: {n_incorrect} ({100*n_incorrect/n_total:.1f}%)")
    lines.append(f"Average reward: {avg_reward:.3f}")
    lines.append("=" * 60)
    lines.append("")

    # Include failures first (more informative for reflection)
    traces_to_show = []

    # Prioritize failures
    n_failures_to_show = min(len(failures), max_traces * 2 // 3)
    traces_to_show.extend(failures[:n_failures_to_show])

    # Add some successes for contrast
    n_successes_to_show = min(len(successes), max_traces - n_failures_to_show)
    traces_to_show.extend(successes[:n_successes_to_show])

    # Section: Failures
    if failures:
        lines.append("=" * 60)
        lines.append(f"FAILURES ({min(n_failures_to_show, len(failures))} shown)")
        lines.append("=" * 60)
        lines.append("")

        for trace in failures[:n_failures_to_show]:
            lines.append(format_single_trace(trace, include_answer=include_answers))
            lines.append("")

    # Section: Successes (for contrast)
    if successes and n_successes_to_show > 0:
        lines.append("=" * 60)
        lines.append(f"SUCCESSES ({n_successes_to_show} shown for contrast)")
        lines.append("=" * 60)
        lines.append("")

        for trace in successes[:n_successes_to_show]:
            lines.append(format_single_trace(trace, include_answer=include_answers))
            lines.append("")

    return "\n".join(lines)


def save_traces(traces: list[ExecutionTrace], path: str) -> None:
    """Save traces to JSONL file."""
    with open(path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict()) + "\n")


def load_traces(path: str) -> list[ExecutionTrace]:
    """Load traces from JSONL file."""
    traces = []
    with open(path) as f:
        for line in f:
            if line.strip():
                traces.append(ExecutionTrace.from_dict(json.loads(line)))
    return traces
