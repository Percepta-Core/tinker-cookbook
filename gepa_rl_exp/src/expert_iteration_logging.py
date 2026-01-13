"""
Logging utilities for expert iteration.

Provides structured logging for debugging and analysis of the expert iteration
pipeline, including (x, y0, hint, y*) examples.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from tinker_cookbook.utils import logtree

from .expert_iteration import ExpertIterationResult


@dataclass
class ExpertIterationLogEntry:
    """Single logged example for expert iteration."""

    batch_idx: int
    problem_idx: int
    question: str
    ground_truth: str
    initial_response: str  # y0
    initial_reward: float
    hint: str | None
    hint_safeguard_applied: bool
    best_response: str  # y*
    best_reward: float
    improvement: float
    num_candidates: int
    used_hint: bool


class ExpertIterationLogger:
    """Logs expert iteration examples to file for debugging."""

    def __init__(self, log_path: str, max_examples_per_batch: int = 5):
        """
        Initialize logger.

        Args:
            log_path: Directory to save logs
            max_examples_per_batch: Maximum examples to log per batch (to limit file size)
        """
        self.log_path = log_path
        self.max_examples_per_batch = max_examples_per_batch
        self.entries: list[ExpertIterationLogEntry] = []

        # Ensure log directory exists
        os.makedirs(log_path, exist_ok=True)

        # Log file path
        self.jsonl_path = os.path.join(log_path, "expert_iteration_examples.jsonl")

    def log_result(
        self,
        result: ExpertIterationResult,
        batch_idx: int,
        problem_idx: int,
    ) -> None:
        """
        Log a single expert iteration result.

        Args:
            result: The ExpertIterationResult to log
            batch_idx: Current batch index
            problem_idx: Problem index within batch
        """
        entry = ExpertIterationLogEntry(
            batch_idx=batch_idx,
            problem_idx=problem_idx,
            question=result.question,
            ground_truth=result.ground_truth,
            initial_response=result.initial_response,
            initial_reward=result.initial_reward,
            hint=result.hint.hint if result.hint else None,
            hint_safeguard_applied=result.hint.safeguard_applied if result.hint else False,
            best_response=result.best_candidate,
            best_reward=result.best_reward,
            improvement=result.improvement,
            num_candidates=len(result.candidates),
            used_hint=result.used_hint,
        )
        self.entries.append(entry)

    def log_batch(
        self,
        results: list[ExpertIterationResult],
        batch_idx: int,
    ) -> None:
        """
        Log a batch of results, limited to max_examples_per_batch.

        Args:
            results: List of ExpertIterationResult
            batch_idx: Current batch index
        """
        # Sort by improvement to log most interesting examples
        sorted_results = sorted(results, key=lambda r: r.improvement, reverse=True)

        # Log up to max_examples_per_batch
        for i, result in enumerate(sorted_results[: self.max_examples_per_batch]):
            self.log_result(result, batch_idx, i)

    def save(self) -> None:
        """Save all entries to JSONL file."""
        with open(self.jsonl_path, "a") as f:
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        self.entries.clear()  # Clear after saving

    def save_summary(self, metrics: dict[str, Any]) -> None:
        """Save a summary of the current batch."""
        summary_path = os.path.join(self.log_path, "expert_iteration_summary.jsonl")
        with open(summary_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def to_logtree(self, result: ExpertIterationResult, batch_idx: int) -> None:
        """Output a single result to logtree HTML format."""
        with logtree.scope_header(f"Expert Iteration Batch {batch_idx}"):
            logtree.log_text(f"Question: {result.question}")
            logtree.log_text(f"Ground Truth: {result.ground_truth}")
            logtree.log_text("")
            logtree.log_text(f"Initial Response (y0): {result.initial_response}")
            logtree.log_text(f"Initial Reward: {result.initial_reward:.3f}")
            logtree.log_text("")
            if result.hint:
                logtree.log_text(f"Hint: {result.hint.hint}")
                if result.hint.safeguard_applied:
                    logtree.log_text(f"  (Safeguard applied: {result.hint.safeguard_reason})")
            else:
                logtree.log_text("Hint: None (not used)")
            logtree.log_text("")
            logtree.log_text(f"Best Response (y*): {result.best_candidate}")
            logtree.log_text(f"Best Reward: {result.best_reward:.3f}")
            logtree.log_text(f"Improvement: {result.improvement:+.3f}")
            logtree.log_text(f"Candidates sampled: {len(result.candidates)}")


def compute_batch_metrics(results: list[ExpertIterationResult]) -> dict[str, float]:
    """
    Compute aggregate metrics for a batch of results.

    Returns:
        Dictionary of metric name -> value
    """
    if not results:
        return {}

    n = len(results)
    n_with_hint = sum(1 for r in results if r.used_hint)
    n_improved = sum(1 for r in results if r.improvement > 0)
    n_safeguard_applied = sum(
        1 for r in results if r.hint and r.hint.safeguard_applied
    )

    return {
        "expert_iter/n_problems": n,
        "expert_iter/n_with_hint": n_with_hint,
        "expert_iter/hint_fraction": n_with_hint / n if n > 0 else 0,
        "expert_iter/n_improved": n_improved,
        "expert_iter/improvement_rate": n_improved / n if n > 0 else 0,
        "expert_iter/mean_initial_reward": sum(r.initial_reward for r in results) / n,
        "expert_iter/mean_best_reward": sum(r.best_reward for r in results) / n,
        "expert_iter/mean_improvement": sum(r.improvement for r in results) / n,
        "expert_iter/n_safeguard_applied": n_safeguard_applied,
        "expert_iter/safeguard_rate": n_safeguard_applied / n_with_hint if n_with_hint > 0 else 0,
    }
