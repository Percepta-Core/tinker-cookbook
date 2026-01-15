"""
GEPA Optimizer: Trace-based reflective prompt evolution.

This module implements the core GEPA optimization loop:
1. Collect execution traces with current prompt
2. Use reflection LM to analyze traces and propose improvements
3. Evaluate proposed prompts
4. Update candidate pool (Pareto frontier)

The key innovation is using LLM reflection on actual execution traces
rather than blind search over prompt templates.
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.metric_util import dataset_to_env_group_builders
from tinker_cookbook.rl.rollouts import do_group_rollout

from .config import GEPAConfig, GEPACandidateMetadata
from .reflection import PromptProposal, ReflectionClient, reflect_on_traces
from .trace import ExecutionTrace, save_traces


@dataclass
class GEPACandidate:
    """
    A candidate prompt in the GEPA optimization pool.

    Tracks the prompt text, evaluation scores, and metadata about
    how it was generated.
    """

    prompt_text: str
    metadata: GEPACandidateMetadata
    scores: list[float] = field(default_factory=list)  # Evaluation scores
    best_score: float = 0.0
    avg_score: float = 0.0

    @property
    def candidate_id(self) -> str:
        return self.metadata.candidate_id

    def add_score(self, score: float) -> None:
        """Record a new evaluation score."""
        self.scores.append(score)
        self.best_score = max(self.scores)
        self.avg_score = sum(self.scores) / len(self.scores)

    def to_dict(self) -> dict:
        return {
            "prompt_text": self.prompt_text,
            "metadata": self.metadata.to_dict(),
            "scores": self.scores,
            "best_score": self.best_score,
            "avg_score": self.avg_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GEPACandidate":
        candidate = cls(
            prompt_text=data["prompt_text"],
            metadata=GEPACandidateMetadata.from_dict(data["metadata"]),
            scores=data.get("scores", []),
        )
        candidate.best_score = data.get("best_score", 0.0)
        candidate.avg_score = data.get("avg_score", 0.0)
        return candidate


class CandidatePool:
    """
    Pool of candidate prompts for GEPA optimization.

    Maintains a Pareto frontier of prompts, keeping the best
    candidates based on evaluation scores.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.candidates: list[GEPACandidate] = []
        self._best_candidate: GEPACandidate | None = None

    def add(self, candidate: GEPACandidate) -> bool:
        """
        Add a candidate to the pool.

        Returns True if the candidate was added (not dominated).
        """
        self.candidates.append(candidate)

        # Update best candidate
        if self._best_candidate is None or candidate.avg_score > self._best_candidate.avg_score:
            self._best_candidate = candidate

        # Prune if over capacity (keep best by avg_score)
        if len(self.candidates) > self.max_size:
            self.candidates.sort(key=lambda c: c.avg_score, reverse=True)
            self.candidates = self.candidates[: self.max_size]

        return candidate in self.candidates

    def get_best(self) -> GEPACandidate | None:
        """Get the best candidate by average score."""
        return self._best_candidate

    def get_current(self) -> GEPACandidate | None:
        """Get the most recent candidate (for iteration)."""
        return self.candidates[-1] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidatePool":
        pool = cls(max_size=data.get("max_size", 5))
        for c_data in data.get("candidates", []):
            pool.add(GEPACandidate.from_dict(c_data))
        return pool


class GEPAOptimizer:
    """
    Main GEPA optimization class.

    Implements the trace-based reflective prompt evolution algorithm:
    1. Collect traces with current prompt
    2. Reflect on traces to diagnose failures
    3. Propose improved prompt
    4. Evaluate and update pool
    """

    def __init__(
        self,
        config: GEPAConfig,
        reflection_client: ReflectionClient,
        dataset_builder_factory: Callable,
        sampling_client: tinker.SamplingClient,
        log_path: str | None = None,
    ):
        """
        Initialize the GEPA optimizer.

        Args:
            config: GEPA configuration
            reflection_client: Client for calling the reflection LM
            dataset_builder_factory: Factory function that takes (prompt) and returns dataset builder
            sampling_client: Tinker sampling client for rollouts
            log_path: Path for saving logs and traces
        """
        self.config = config
        self.reflection_client = reflection_client
        self.dataset_builder_factory = dataset_builder_factory
        self.sampling_client = sampling_client
        self.log_path = log_path

        # State
        self.pool = CandidatePool(max_size=config.pool_size)
        self.iteration = 0
        self.all_traces: list[ExecutionTrace] = []

        # Initialize with seed prompt
        seed_prompt = config.seed_prompt or self._default_seed_prompt()
        seed_candidate = GEPACandidate(
            prompt_text=seed_prompt,
            metadata=GEPACandidateMetadata(
                candidate_id=self._generate_id(),
                parent_id=None,
                iteration_created=0,
                proposal_reasoning="Seed prompt",
            ),
        )
        self.pool.add(seed_candidate)

    def _default_seed_prompt(self) -> str:
        """Default seed prompt if none provided."""
        return "Solve the problem step by step. Show your reasoning clearly."

    def _generate_id(self) -> str:
        """Generate a unique candidate ID."""
        return f"c_{uuid.uuid4().hex[:8]}"

    async def collect_traces(
        self,
        prompt: str,
        n_problems: int,
    ) -> list[ExecutionTrace]:
        """
        Collect execution traces for a given prompt.

        Runs rollouts on a sample of problems and captures full traces
        including model outputs, rewards, and any env feedback.

        Args:
            prompt: The prompt to use for rollouts
            n_problems: Number of problems to sample

        Returns:
            List of ExecutionTrace objects
        """
        # Build dataset with the given prompt
        dataset_builder = self.dataset_builder_factory(prompt)
        train_dataset, _ = await dataset_builder()

        # Get env builders
        all_env_builders = dataset_to_env_group_builders(train_dataset)
        env_builders = all_env_builders[:n_problems]

        # Create policy
        policy = TinkerTokenCompleter(
            sampling_client=self.sampling_client,
            max_tokens=256,  # Could make configurable
            temperature=1.0,
        )

        # Run rollouts
        trajectory_groups = await asyncio.gather(
            *[do_group_rollout(builder, policy) for builder in env_builders]
        )

        # Convert trajectories to traces
        traces: list[ExecutionTrace] = []

        for i, (builder, traj_group) in enumerate(zip(env_builders, trajectory_groups)):
            for j, traj in enumerate(traj_group.trajectories_G):
                # Extract info from trajectory
                episode_reward = sum(t.reward for t in traj.transitions)

                # Get correctness from metrics
                correct = False
                for t in traj.transitions:
                    if "correct" in t.metrics:
                        correct = bool(t.metrics["correct"])
                        break

                # Get model output (concatenate all action tokens)
                output_parts = []
                for t in traj.transitions:
                    if hasattr(t.ac, "tokens"):
                        # Decode tokens if possible, otherwise use raw
                        output_parts.append(str(t.ac.tokens))

                # Get env feedback from metrics
                env_feedback = None
                for t in traj.transitions:
                    if "feedback" in t.metrics:
                        env_feedback = str(t.metrics["feedback"])
                    elif "error" in t.metrics:
                        env_feedback = str(t.metrics["error"])

                # Get expected answer if available
                expected_answer = None
                if hasattr(builder, "answer"):
                    expected_answer = str(builder.answer)

                # Get input text
                input_text = getattr(builder, "question", f"Problem {i}")

                trace = ExecutionTrace(
                    problem_id=f"p{i}_r{j}",
                    input_text=input_text,
                    system_prompt=prompt,
                    output=" ".join(output_parts),
                    reward=episode_reward,
                    correct=correct,
                    env_feedback=env_feedback,
                    expected_answer=expected_answer,
                )
                traces.append(trace)

        return traces

    async def propose_improvement(
        self,
        traces: list[ExecutionTrace],
        current_prompt: str,
    ) -> PromptProposal:
        """
        Use reflection LM to propose a prompt improvement.

        Args:
            traces: Execution traces to analyze
            current_prompt: The current prompt being used

        Returns:
            PromptProposal with the suggested improvement
        """
        return await reflect_on_traces(traces, current_prompt, self.reflection_client)

    async def evaluate_prompt(
        self,
        prompt: str,
        n_problems: int,
    ) -> float:
        """
        Evaluate a prompt on a sample of problems.

        Returns the average reward (correctness rate for binary rewards).
        """
        traces = await self.collect_traces(prompt, n_problems)
        if not traces:
            return 0.0

        return sum(t.reward for t in traces) / len(traces)

    async def step(self) -> GEPACandidate:
        """
        Execute one GEPA optimization step.

        1. Collect traces with current best prompt
        2. Reflect on traces to propose improvement
        3. Evaluate proposed prompt
        4. Update candidate pool

        Returns:
            The new candidate (whether accepted or not)
        """
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"GEPA Iteration {self.iteration}")
        print(f"{'='*60}")

        # Get current best prompt
        current = self.pool.get_best()
        if current is None:
            raise RuntimeError("No candidates in pool")

        current_prompt = current.prompt_text
        print(f"Current prompt: {current_prompt[:100]}...")

        # 1. Collect traces
        print(f"\nCollecting traces ({self.config.traces_per_iteration} problems)...")
        traces = await self.collect_traces(
            current_prompt,
            self.config.traces_per_iteration,
        )
        self.all_traces.extend(traces)

        # Log traces
        if self.config.log_traces and self.log_path:
            trace_path = os.path.join(self.log_path, f"traces_iter{self.iteration}.jsonl")
            save_traces(traces, trace_path)
            print(f"Saved traces to {trace_path}")

        # Compute current performance
        n_correct = sum(1 for t in traces if t.correct)
        current_score = n_correct / len(traces) if traces else 0.0
        print(f"Current performance: {n_correct}/{len(traces)} = {current_score:.1%}")

        # 2. Reflect on traces
        print("\nReflecting on traces...")
        proposal = await self.propose_improvement(traces, current_prompt)
        print(f"Failure analysis: {proposal.failure_analysis}")
        print(f"Failure patterns: {proposal.failure_patterns}")
        print(f"Proposed prompt: {proposal.new_prompt[:100]}...")
        print(f"Reasoning: {proposal.reasoning}")

        # Log proposal
        if self.config.log_proposals and self.log_path:
            proposal_path = os.path.join(self.log_path, f"proposal_iter{self.iteration}.json")
            with open(proposal_path, "w") as f:
                json.dump(proposal.to_dict(), f, indent=2)

        # 3. Evaluate proposed prompt
        print(f"\nEvaluating proposed prompt ({self.config.traces_per_iteration} problems)...")
        new_traces = await self.collect_traces(
            proposal.new_prompt,
            self.config.traces_per_iteration,
        )

        n_new_correct = sum(1 for t in new_traces if t.correct)
        new_score = n_new_correct / len(new_traces) if new_traces else 0.0
        print(f"New performance: {n_new_correct}/{len(new_traces)} = {new_score:.1%}")

        # 4. Create and score new candidate
        new_candidate = GEPACandidate(
            prompt_text=proposal.new_prompt,
            metadata=GEPACandidateMetadata(
                candidate_id=self._generate_id(),
                parent_id=current.candidate_id,
                iteration_created=self.iteration,
                proposal_reasoning=proposal.reasoning,
                failure_patterns=proposal.failure_patterns,
            ),
        )
        new_candidate.add_score(new_score)

        # Also update current candidate's score
        current.add_score(current_score)

        # 5. Update pool
        improvement = new_score - current_score
        if improvement >= self.config.min_improvement_threshold:
            self.pool.add(new_candidate)
            print(f"\nACCEPTED: Improvement of {improvement:.1%}")
        else:
            print(f"\nREJECTED: Improvement of {improvement:.1%} < threshold {self.config.min_improvement_threshold:.1%}")

        return new_candidate

    async def optimize(self, n_iterations: int | None = None) -> GEPACandidate:
        """
        Run the full GEPA optimization loop.

        Args:
            n_iterations: Number of iterations (defaults to config.max_iterations)

        Returns:
            The best candidate found
        """
        n_iter = n_iterations or self.config.max_iterations

        for _ in range(n_iter):
            await self.step()

        best = self.pool.get_best()
        if best is None:
            raise RuntimeError("No candidates found")

        print(f"\n{'='*60}")
        print("GEPA Optimization Complete")
        print(f"{'='*60}")
        print(f"Best prompt: {best.prompt_text}")
        print(f"Best score: {best.best_score:.1%}")
        print(f"Total iterations: {self.iteration}")

        return best

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for checkpointing."""
        return {
            "iteration": self.iteration,
            "pool": self.pool.to_dict(),
            "config": {
                "max_iterations": self.config.max_iterations,
                "traces_per_iteration": self.config.traces_per_iteration,
                "pool_size": self.config.pool_size,
            },
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.iteration = state.get("iteration", 0)
        if "pool" in state:
            self.pool = CandidatePool.from_dict(state["pool"])
