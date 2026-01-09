# Math RL Training Wrapper

A wrapper around `tinker-cookbook` for running math RL training with customizable prompt templates.

## Features

- **Prompt Templates**: Three built-in templates (baseline, step-by-step, concise) for different reasoning styles
- **Organized Outputs**: All runs saved to `runs/` with auto-generated names
- **Shell Scripts**: Ready-to-use scripts for smoke tests, training, and evaluation
- **GSM8K & MATH Support**: Works with both grade-school and competition math datasets

## Installation

### Prerequisites

1. **Tinker API Access**: Sign up at [thinkingmachines.ai/tinker](https://thinkingmachines.ai/tinker) and get your API key

2. **Set environment variable**:
   ```bash
   export TINKER_API_KEY=sk-...
   ```

### Setup

```bash
# 1. Install tinker-cookbook (from parent directory)
cd ../tinker-cookbook
pip install -e .[dev]

# 2. Install this wrapper
cd ../math-rl-wrapper
pip install -e .
```

## Quick Start

### Smoke Test (~2-5 minutes)

Quick validation that everything works:

```bash
./scripts/smoke_test.sh
```

With a specific template:

```bash
./scripts/smoke_test.sh step_by_step
```

### Full Training

Train on GSM8K with baseline template:

```bash
./scripts/train_math_rl.sh gsm8k baseline
```

Train on MATH with step-by-step reasoning:

```bash
./scripts/train_math_rl.sh math step_by_step meta-llama/Llama-3.1-8B-Instruct
```

### Direct Python Invocation

For full control over all parameters:

```bash
python -m src.train \
    env=gsm8k \
    prompt_template=step_by_step \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    groups_per_batch=100 \
    group_size=4 \
    learning_rate=1e-5 \
    max_tokens=256 \
    wandb_project=math-rl-experiments
```

### Evaluation

Evaluate a trained checkpoint:

```bash
./scripts/eval_math_rl.sh tinker://path/to/checkpoint
```

## Prompt Templates

Three templates are available, each demonstrating a different reasoning style:

| Template | Description | Best For |
|----------|-------------|----------|
| `baseline` | Simple enumeration (default) | General math problems |
| `step_by_step` | Verbose, labeled steps | Complex multi-step problems |
| `concise` | Minimal reasoning | Simple calculations, speed |

Use `prompt_template=none` to disable few-shot examples entirely.

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env` | `gsm8k` | Dataset: `gsm8k` or `math` |
| `prompt_template` | `baseline` | Template: `baseline`, `step_by_step`, `concise`, `none` |
| `model_name` | `meta-llama/Llama-3.1-8B-Instruct` | Base model to fine-tune |
| `groups_per_batch` | 100 | Number of problems per batch |
| `group_size` | 4 | Rollouts per problem (for GRPO) |
| `learning_rate` | 1e-5 | Learning rate |
| `max_tokens` | 256 | Max generation length |
| `lora_rank` | 32 | LoRA rank |
| `eval_every` | 20 | Evaluation frequency (batches) |
| `save_every` | 20 | Checkpoint frequency (batches) |
| `wandb_project` | None | W&B project name |
| `log_path` | `runs/<auto>` | Output directory |

## Output Files

Each training run creates:

```
runs/<run_name>/
├── config.json          # Full configuration
├── metrics.jsonl        # Training metrics (one JSON per line)
├── checkpoints.jsonl    # Checkpoint metadata
└── *.html               # Trajectory visualizations (if enabled)
```

### Viewing Metrics

```bash
# Latest metrics
cat runs/<run_name>/metrics.jsonl | tail -5 | jq .

# Reward over time
cat runs/<run_name>/metrics.jsonl | jq -r '.["env/all/reward/total"] // empty'
```

### Checkpoints

Checkpoints are saved to Tinker's cloud storage. Find paths in:

```bash
cat runs/<run_name>/checkpoints.jsonl | jq .
```

## Future Extensions

This wrapper is designed to support:

1. **GEPA-style Prompt Search**: Outer loop over `prompt_template` values
2. **Expert Iteration**: Use successful traces for SFT distillation
3. **Custom Templates**: Add new templates in `src/prompt_templates.py`

## Troubleshooting

### API Authentication

```
Error: Authentication failed
```

Ensure `TINKER_API_KEY` is set:
```bash
echo $TINKER_API_KEY  # Should show sk-...
```

### Dataset Download

First run downloads datasets from HuggingFace. Ensure you have internet access and sufficient disk space.

### Log Directory Exists

If a previous run used the same log path:
```bash
# Delete and retry
python -m src.train ... behavior_if_log_dir_exists=delete

# Or resume from checkpoint
python -m src.train ... behavior_if_log_dir_exists=resume
```
