"""
GEPA: Trace-based reflective prompt evolution.

This module implements the GEPA algorithm (arxiv:2507.19457) for optimizing
prompts by analyzing execution traces and using LLM reflection to propose
targeted improvements.
"""

from .trace import ExecutionTrace, format_traces_for_reflection
from .reflection import PromptProposal, reflect_on_traces
from .optimizer import GEPAOptimizer, GEPACandidate
from .config import GEPAConfig

__all__ = [
    "ExecutionTrace",
    "format_traces_for_reflection",
    "PromptProposal",
    "reflect_on_traces",
    "GEPAOptimizer",
    "GEPACandidate",
    "GEPAConfig",
]
