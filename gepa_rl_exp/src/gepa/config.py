"""
Configuration for GEPA optimization.

This module defines the configuration dataclass for the GEPA optimizer,
including settings for the reflection model, optimization budget, and
candidate pool management.
"""

from dataclasses import dataclass, field
from typing import Literal

import chz


@chz.chz
class GEPAConfig:
    """
    Configuration for GEPA trace-based prompt optimization.

    GEPA uses LLM reflection on execution traces to iteratively improve
    prompts. This config controls the reflection model, optimization
    budget, and candidate pool management.
    """

    # Reflection model settings
    reflection_backend: Literal["openai", "anthropic"] = "anthropic"
    reflection_model: str = "claude-opus-4-5-20251101"  # Model for analyzing traces
    reflection_temperature: float = 0.7
    reflection_max_tokens: int = 2000

    # Optimization budget
    max_iterations: int = 10  # Total GEPA optimization steps
    traces_per_iteration: int = 16  # Problems to sample per step
    rollouts_per_problem: int = 4  # Rollouts per problem for variance

    # Candidate pool (Pareto frontier)
    pool_size: int = 5  # Max candidates to maintain
    min_improvement_threshold: float = 0.01  # Min reward improvement to accept

    # Seed prompt
    seed_prompt: str | None = None  # Starting prompt (uses baseline if None)

    # Evaluation
    eval_problems: int = 32  # Problems for final evaluation
    eval_rollouts: int = 4  # Rollouts per problem for evaluation

    # Logging
    log_traces: bool = True  # Save traces to disk
    log_proposals: bool = True  # Save proposals to disk

    # Reproducibility
    seed: int = 0


@dataclass
class GEPACandidateMetadata:
    """Metadata for a GEPA candidate prompt."""

    candidate_id: str
    parent_id: str | None
    iteration_created: int
    proposal_reasoning: str
    failure_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "iteration_created": self.iteration_created,
            "proposal_reasoning": self.proposal_reasoning,
            "failure_patterns": self.failure_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GEPACandidateMetadata":
        return cls(
            candidate_id=data["candidate_id"],
            parent_id=data.get("parent_id"),
            iteration_created=data.get("iteration_created", 0),
            proposal_reasoning=data.get("proposal_reasoning", ""),
            failure_patterns=data.get("failure_patterns", []),
        )
