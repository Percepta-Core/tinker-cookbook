# GEPA + RL Experiments

A wrapper around `tinker-cookbook` for running GEPA (Guided Evolution for Prompt Adaptation) combined with RL training on math datasets.

## Overview

This project implements **block coordinate ascent** on prompts and weights:
- **GEPA Phase**: Optimize the system prompt using trace-based LLM reflection
- **RL Phase**: Train model weights with the optimized prompt held fixed

The GEPA implementation follows [arxiv:2507.19457](https://arxiv.org/abs/2507.19457), using an external LLM (Claude/GPT-4) to analyze execution traces and propose prompt improvements.

## Features

- **True GEPA**: Trace-based reflective prompt optimization (not just template selection)
- **GEPA+RL Alternation**: Block coordinate ascent between prompt optimization and weight training
- **Multiple Datasets**: GSM8K, MATH, and arithmetic (toy environment for testing)
- **W&B Integration**: Track experiments with Weights & Biases

## Installation

### Prerequisites

1. **Tinker API Access**: Get your API key from [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker)

2. **Set environment variables**:
   ```bash
   export TINKER_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...  # For GEPA reflection (or OPENAI_API_KEY)
   ```

### Setup

```bash
# Install tinker-cookbook (from parent directory)
cd ../tinker-cookbook
pip install -e .[dev]

# Install this wrapper
cd ../gepa_rl_exp
pip install -e .
```

## Quick Start

### Smoke Test (~2-5 minutes)

Quick validation on arithmetic environment:

```bash
./scripts/smoke_test.sh
```

### RL-Only Training

Train on GSM8K with pure RL (no GEPA):

```bash
python -m src.train \
    env=gsm8k \
    n_batches=20 \
    learning_rate=5e-4 \
    wandb_project=gepa-rl
```

### GEPA + RL Training

Run alternating GEPA and RL phases:

```bash
python -m scripts.run_alternation \
    env=gsm8k \
    n_rounds=2 \
    rl_batches_per_round=10 \
    skip_first_gepa=true \
    reflection_provider=anthropic \
    reflection_model=claude-opus-4-5-20251101 \
    wandb_project=gepa-rl
```

Key options:
- `skip_first_gepa=true`: Start with RL training (recommended - no traces to analyze initially)
- `n_rounds`: Number of GEPA→RL cycles
- `rl_batches_per_round`: RL training batches between GEPA phases

## Architecture

### GEPA Module (`src/gepa/`)

```
src/gepa/
├── trace.py       # ExecutionTrace dataclass for rollout records
├── reflection.py  # LLM-based trace analysis and prompt proposals
├── optimizer.py   # GEPAOptimizer manages optimization loop
└── config.py      # GEPAConfig settings
```

**How GEPA works:**
1. Collect execution traces (input, output, reward) from model rollouts
2. Send traces to reflection LLM (Claude/GPT-4) for analysis
3. Reflection LLM diagnoses failure patterns and proposes prompt improvements
4. Evaluate proposed prompt on a probe batch
5. Accept if improvement, otherwise keep current prompt

### Scripts

| Script | Description |
|--------|-------------|
| `src/train.py` | Pure RL training |
| `scripts/run_alternation.py` | GEPA+RL block coordinate ascent |
| `scripts/run_gepa.py` | GEPA-only optimization (for testing) |

## Configuration

### RL Training (`src/train.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env` | `gsm8k` | Dataset: `arithmetic`, `gsm8k`, or `math` |
| `model_name` | `Qwen/Qwen3-8B` | Base model to fine-tune |
| `n_batches` | `None` | Limit batches (None = full dataset) |
| `groups_per_batch` | 50 | Problems per batch |
| `group_size` | 4 | Rollouts per problem |
| `learning_rate` | `5e-4` | Learning rate (LoRA default) |
| `lora_rank` | 32 | LoRA rank |
| `max_tokens` | 256 | Max generation length |
| `wandb_project` | `None` | W&B project name |

### GEPA+RL (`scripts/run_alternation.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rounds` | 3 | Number of GEPA→RL cycles |
| `rl_batches_per_round` | 10 | RL batches per round |
| `skip_first_gepa` | `false` | Skip GEPA in first round |
| `reflection_provider` | `anthropic` | LLM provider for reflection |
| `reflection_model` | `claude-opus-4-5-20251101` | Model for trace analysis |
| `gepa_iterations` | 3 | GEPA optimization iterations per phase |
| `traces_per_iteration` | 16 | Problems sampled per GEPA iteration |

## Experiment Results

GSM8K experiments with Qwen3-8B, 20 batches, LR=5e-4:

| Experiment | Initial | Final Test Accuracy |
|------------|---------|---------------------|
| RL-only | 43.5% | 83.5% |
| GEPA-midway (GEPA at step 10) | 43.8% | 83.8% |
| GEPA-twice (GEPA at steps 7, 14) | 44.0% | 87.3% |

**Key findings:**
- RL training works well with correct LoRA LR (5e-4, not 1e-5)
- Most improvement happens in first 3-5 batches
- GEPA shows marginal benefit over pure RL in these experiments

## Output Files

```
runs/<run_name>/
├── config.json          # Full configuration
├── metrics.jsonl        # Training metrics
├── checkpoints.jsonl    # Checkpoint metadata
├── round0_rl/           # Per-round RL metrics (alternation mode)
├── round1_rl/
└── *.html               # Trajectory visualizations
```

### Viewing Metrics

```bash
# Key metrics per batch
cat runs/<run_name>/metrics.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    print(f\"batch {d['progress/batch']:2d}: correct={d['env/all/correct']:.0%} reward={d['env/all/reward/total']:.2f}\")
"
```

## Troubleshooting

### Learning Rate

For LoRA training, use `learning_rate=5e-4` (not 1e-5). The recommended LoRA LR is ~50x higher than full fine-tuning.

### GEPA Reflection Errors

Ensure your API key is set for the reflection provider:
```bash
export ANTHROPIC_API_KEY=sk-ant-...  # For Anthropic
export OPENAI_API_KEY=sk-...         # For OpenAI
```

### Format Regression

If GEPA causes format degradation (model outputs correct answers but wrong format), consider:
- Adding format compliance to the reflection prompt
- Using `skip_first_gepa=true` to establish good formatting first
- Reducing `gepa_iterations` to limit prompt changes
