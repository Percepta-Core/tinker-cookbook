# claude.md

This repo implements RL with verifiable rewards (RLVR) and extensions inspired by the "GEPARL" note (Jan 2026): joint GEPA + RL (optimize a **global harness** `s` + weights `θ`) and expert iteration (use **extra compute** at training time to generate better targets and distill into `θ`).

You (Claude Code) will be asked to implement features in this codebase. Please follow the constraints and mental model below.

---

## Goals

1. **Keep RLVR working end-to-end** (existing `math_rl` recipe currently runs).
2. Add **GEPA + RL** alternation:
   - Optimize a **global harness** `s` (e.g., system prompt / message template / output schema).
   - Update weights `θ` with RL while holding `s` fixed.
3. Add **expert iteration** (outer loop), using **teacher-generated hints**:
   - Student generates an attempt.
   - A strong teacher model compares attempt vs ground truth and produces a **non-leaking hint**.
   - Use hint only inside expert-time search/refinement to produce a better output.
   - Distill the improved output into the student **under the clean global harness**.

All additions must be **dataset-agnostic** (should work for math_rl and other tasks like TauBench).

---

## Notation (for internal reasoning)

- Dataset provides examples `x` and ground-truth answers (format varies by dataset).
- Policy: `p(y | x; s, θ)` where:
  - `θ` = model weights
  - `s` = harness/template/prompt program (global unless explicitly stated)
- Verifiable reward: `R(x, y)` computed by a deterministic or near-deterministic checker (often binary 0/1).
- Objective: `J(s, θ) = E[R(x, y)]`.

**Important:** This project is *not* preference optimization. No pairwise comparisons for reward. Reward comes from `R(x,y)`.

---

## Invariants (DO NOT BREAK)

### Dataset-agnostic design
- Do **not** assume math formatting (no `\boxed{}`, `####`, etc.).
- Treat dataset example `x` as an opaque record; only the dataset adapter knows how to stringify it.
- Reward is always via `R(x, y)`; do not introduce LLM-as-judge reward.

### Harness / template semantics
- `s` is the **global harness** used by the student at train and eval time (unless explicitly doing an ablation).
- GEPA (policy-side) searches over **global** `s` only.
- Expert iteration may use **problem-specific hints** but **only inside expert search**; the student does **not** see hints during distillation.

### Expert iteration with teacher hints
- Teacher hint generation must:
  - Use the dataset's ground truth (whatever field exists) + the student attempt.
  - Avoid leaking the final answer verbatim.
- Distillation trains `θ` on `(x -> y*)` under the clean global `s`.

---

## Implementation Principles

### Keep changes minimal and local
- Prefer adding small modules and plumbing them into existing trainers rather than rewriting training loops.
- Avoid new heavy dependencies. Use standard library where possible.

### Reproducibility
- Seed all RNG used for GEPA proposal sampling and any stochastic expert search.
- Log configuration and seeds at startup.

### Checkpoint/resume
- Any stateful addition must be checkpointable:
  - Current `s`
  - PromptPool state (if implementing GEPA)
  - Schedule counters (when GEPA ran last, etc.)
  - Schedule counters (when GEPA ran last, etc.)
  - Any expert-iteration buffers (if persistent)

### Logging
- Always log:
  - current mode (RL-only, GEPA+RL, expert-iteration enabled)
  - reward stats
  - if `s` changes: old id/new id and a short excerpt
  - sample debug artifacts for teacher hints (a small rotating sample set saved to disk)

---

## Components to Build

### 1) Harness abstraction (global `s`)
Refactor so the harness/template is a first-class object that can be swapped without touching weights.
- Single source of truth for `s` in run state/config.
- Both rollouts and evaluation must use the same `s`.

Suggested interface (adapt to repo style):
- `build_messages(x, s) -> messages`
- Optional: `postprocess_output(y_raw, s) -> y`

### 2) GEPA + RL (policy-side prompt optimization)
Implement the alternation:
- Many RL steps with fixed `s`.
- Occasionally:
  - Propose candidate `s_k` (from a pool + mutations/recombination).
  - Estimate `J(s_k, θ)` on a probe set (cheap).
  - Set `s <- argmax_k J(s_k, θ)`.

Do not hard-code math heuristics; `J` must use `R(x,y)` only.

**PromptPool**
- Stores K candidates `s` plus metadata (id, parents, operator, last score).
- JSON serialize for checkpoints.

**Mutations**
- Deterministic/seeded edits and recombination.
- No external LLM calls for mutation generation.

### 3) Expert iteration (teacher hint → expert search → distill)
This is separate from GEPA+RL and can be enabled independently.

**High-level steps for each x:**
1. Student attempt: `y0 ~ p(.|x; s, θ)`
2. Teacher hint: strong model takes `(x, y0, gt)` and outputs a short hint `h`:
   - Must not reveal the ground truth answer.
3. Expert search/refinement: run student with hint **only inside expert** to generate candidates.
4. Score candidates with `R(x,y)`; select `y*`.
5. Distill: train student on `log p(y* | x; s, θ)` (optionally weighted).

**Teacher model integration**
- Keep it behind a simple interface so provider can be swapped.
- Prompt template must be generic and enforce non-leak constraints.

**Hint leakage safeguards**
- Reject hints containing exact substrings of the ground-truth answer (when representable as a string).
- Consider simple heuristics (length limits, ban "the answer is", ban long numeric tokens) but keep it dataset-agnostic.

---

## Configuration / Ablations (must support)
Add config flags so we can run experiments without code edits:

- `mode: RL_ONLY | GEPA_RL`
- `enable_expert_iteration: bool`
- `enable_teacher_hints: bool` (if expert iteration enabled)
- GEPA knobs: `K`, `probe_batch_size`, `probe_rollouts_per_x`, `gepa_period_steps`, etc.
- Expert knobs: `expert_samples_per_x`, `fraction_of_batch_with_hints`, hint length limits, teacher model name.

Ablations we care about:
- RL-only with fixed `s`
- GEPA+RL (global `s` changes)
- Expert iteration without hints (best-of-N only)
- Expert iteration with teacher hints
- GEPA+RL + expert iteration


## What to Avoid

- Introducing dataset-specific formatting assumptions in core training logic.
- Using an LLM-as-judge for reward. Teacher is allowed only for *hint text*, never for scoring.
- Changing the deployed policy class to `p(y|x; s(x), θ)` unless explicitly asked; default is fixed global `s`.
- Large refactors unless necessary.

---

## Working Style for Claude Code

When implementing a change:
1. **Identify the exact files and entrypoints** involved.
2. Propose a **minimal patch plan**.
3. Implement in small steps, keeping existing behavior identical when new features are disabled.
4. Add a tiny test (unit/integration) when possible.
5. Summarize:
   - Files changed
   - How to run
   - New config flags
   - Expected logs/output

If anything is ambiguous in the codebase (e.g., dataset field names), make a best effort by inspecting existing adapters and follow established patterns rather than inventing new ones.

---

## Code Review
Before committing code, do a round of critical reflection about the changes:
- Are there any duplicated logic or validation between different layers (e.g., Python and C)?
- Could there be race conditions or inconsistencies between components?
- Is there unnecessary work being done (e.g., redundant scans, repeated computations)?
- Are there potential memory leaks or error paths that don't clean up properly?
- Are parameter/function names clear and accurate?
- Is the fix complete, or does it leave related issues unaddressed?

---
