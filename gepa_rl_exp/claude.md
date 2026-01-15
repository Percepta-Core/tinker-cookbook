# claude.md

This repo implements RL with verifiable rewards (RLVR) and extensions including:
- **GEPA**: Trace-based reflective prompt evolution (arxiv:2507.19457)
- **GEPA + RL**: Joint optimization over prompt `s` and weights `θ`
- **Expert iteration**: Use extra compute at training time with teacher hints

---

## GEPA: Trace-Based Reflective Prompt Evolution

GEPA optimizes prompts by analyzing execution traces and using LLM reflection
to propose targeted improvements. This is fundamentally different from template
selection or grid search.

### Core Algorithm

```
for each optimization step:
    1. Sample minibatch of problems
    2. Run rollouts with current prompt, collect execution traces
    3. Reflection LM analyzes traces → diagnoses failures → proposes new prompt
    4. Evaluate proposed prompt on minibatch
    5. Update candidate pool if improvement exceeds threshold
```

### Key Components

**Execution Traces** (`src/gepa/trace.py`):
- Captures complete rollout trajectory: input, messages, output, reward, feedback
- Serialized to text for the reflection LM to analyze
- Organizes failures and successes for pattern analysis

**Reflection LM** (`src/gepa/reflection.py`):
- External API (GPT-4o/Claude) analyzes traces
- Identifies failure patterns across multiple rollouts
- Proposes concrete, targeted prompt improvements
- Returns structured proposal with reasoning

**Optimizer** (`src/gepa/optimizer.py`):
- `GEPAOptimizer` orchestrates the optimization loop
- `CandidatePool` maintains Pareto frontier of prompts
- Checkpointable state for resume capability

### Usage

```bash
# Basic GEPA optimization
python -m scripts.run_gepa env=arithmetic max_iterations=5

# With specific reflection model
python -m scripts.run_gepa env=gsm8k reflection_model=gpt-4o reflection_backend=openai

# Debug mode (dummy reflection)
python -m scripts.run_gepa env=arithmetic debug=True
```

### Configuration

```python
GEPAConfig:
    reflection_backend: "openai" | "anthropic"  # Reflection LM provider
    reflection_model: str = "gpt-4o"            # Model for trace analysis
    max_iterations: int = 10                    # GEPA optimization steps
    traces_per_iteration: int = 16              # Problems per step
    rollouts_per_problem: int = 4               # For variance estimation
    pool_size: int = 5                          # Max candidates in frontier
    min_improvement_threshold: float = 0.01    # Accept threshold
    seed_prompt: str | None                     # Starting prompt
```

---

## Goals

1. **Keep RLVR working end-to-end** (existing `math_rl` recipe currently runs).
2. **GEPA + RL** alternation:
   - Use trace-based reflection to optimize prompt `s`
   - Update weights `θ` with RL while holding `s` fixed
3. **Expert iteration** (outer loop), using **teacher-generated hints**:
   - Student generates an attempt
   - Teacher model produces a **non-leaking hint**
   - Use hint only inside expert-time search/refinement
   - Distill the improved output into the student **under the clean prompt**

All additions must be **dataset-agnostic** (should work for math_rl and other tasks).

---

## Notation (for internal reasoning)

- Dataset provides examples `x` and ground-truth answers (format varies by dataset)
- Policy: `p(y | x; s, θ)` where:
  - `θ` = model weights
  - `s` = prompt/harness (global unless explicitly stated)
- Verifiable reward: `R(x, y)` computed by a deterministic checker (often binary 0/1)
- Objective: `J(s, θ) = E[R(x, y)]`

**Important:** This project is *not* preference optimization. Reward comes from `R(x,y)`.

---

## Invariants (DO NOT BREAK)

### Dataset-agnostic design
- Do **not** assume math formatting (no `\boxed{}`, `####`, etc.)
- Treat dataset example `x` as an opaque record
- Reward is always via `R(x, y)`; do not introduce LLM-as-judge reward

### GEPA semantics
- GEPA uses **LLM reflection on execution traces** to propose prompts
- NOT template selection or grid search
- The reflection LM receives actual failure examples to diagnose patterns
- Proposals are targeted based on observed failures

### Expert iteration with teacher hints
- Teacher hint generation must:
  - Use the dataset's ground truth + the student attempt
  - Avoid leaking the final answer verbatim
- Distillation trains `θ` on `(x -> y*)` under the clean prompt `s`

---

## Implementation Principles

### Keep changes minimal and local
- Prefer adding small modules rather than rewriting training loops
- Avoid new heavy dependencies

### Reproducibility
- Seed all RNG used for GEPA and expert search
- Log configuration and seeds at startup

### Checkpoint/resume
- Any stateful addition must be checkpointable:
  - Current `s` (prompt text)
  - CandidatePool state
  - Schedule counters
  - Expert-iteration buffers

### Logging
- Always log:
  - current mode (RL-only, GEPA+RL, expert-iteration enabled)
  - reward stats
  - if `s` changes: old/new prompt excerpts
  - sample debug artifacts for traces and proposals

---

## Project Structure

```
gepa_rl_exp/
├── src/
│   ├── train.py              # Main RL training entry point
│   ├── prompt_strategy.py    # PromptStrategy protocol and implementations
│   ├── prompt_templates.py   # Legacy template definitions
│   ├── gepa/                  # GEPA optimizer module
│   │   ├── __init__.py
│   │   ├── config.py         # GEPAConfig dataclass
│   │   ├── trace.py          # ExecutionTrace and serialization
│   │   ├── reflection.py     # ReflectionClient and prompts
│   │   └── optimizer.py      # GEPAOptimizer class
│   ├── expert_iteration.py   # Expert iteration with hints
│   ├── hint_generation.py    # Teacher hint generation
│   └── teacher_client.py     # OpenAI/Anthropic API clients
├── scripts/
│   ├── run_gepa.py           # GEPA optimization script
│   ├── run_alternation.py    # GEPA+RL alternation loop
│   └── run_expert_iteration.py
└── runs/                      # Output directory
```

---

## Quick Commands

```bash
cd gepa_rl_exp

# GEPA optimization only
python -m scripts.run_gepa env=arithmetic max_iterations=5

# GEPA+RL alternation
python -m scripts.run_alternation env=arithmetic n_rounds=3 gepa_iterations=3

# With wandb logging
python -m scripts.run_alternation env=arithmetic wandb_project=gepa-rl

# Debug mode (no external API calls)
python -m scripts.run_gepa env=arithmetic debug=True
```

---

## Configuration / Ablations

Config flags for experiments without code edits:

- `mode: RL_ONLY | GEPA_RL` (via script choice)
- `enable_expert_iteration: bool`
- `enable_teacher_hints: bool`
- GEPA knobs: `max_iterations`, `traces_per_iteration`, `reflection_model`
- Expert knobs: `expert_samples_per_x`, hint length limits, teacher model name

Ablations we care about:
- RL-only with fixed prompt
- GEPA+RL (prompt evolves via reflection)
- Expert iteration without hints (best-of-N only)
- Expert iteration with teacher hints
- GEPA+RL + expert iteration

---

## What to Avoid

- Using template selection for GEPA (use trace-based reflection instead)
- Introducing dataset-specific formatting assumptions in core logic
- Using an LLM-as-judge for reward (teacher is allowed only for hints)
- Large refactors unless necessary

---

## Working Style for Claude Code

When implementing a change:
1. **Identify the exact files and entrypoints** involved
2. Propose a **minimal patch plan**
3. Implement in small steps, keeping existing behavior identical when new features are disabled
4. Add a tiny test (unit/integration) when possible
5. Summarize:
   - Files changed
   - How to run
   - New config flags
   - Expected logs/output

---

## Code Review

Before committing code, critically reflect on:
- Are there duplicated logic or validation between layers?
- Could there be race conditions or inconsistencies?
- Is there unnecessary work being done?
- Are there potential memory leaks or error paths that don't clean up?
- Are parameter/function names clear and accurate?
- Is the fix complete, or does it leave related issues unaddressed?

---
