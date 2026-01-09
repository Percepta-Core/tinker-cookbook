#!/bin/bash
# Full math RL training script
# Supports GSM8K and MATH datasets with customizable templates

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse positional arguments with defaults
ENV="${1:-gsm8k}"
TEMPLATE="${2:-baseline}"
MODEL="${3:-meta-llama/Llama-3.1-8B-Instruct}"

# Generate run name
RUN_NAME="${ENV}_${TEMPLATE}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="runs/${RUN_NAME}"

echo "============================================"
echo "Math RL Training"
echo "============================================"
echo "Environment: $ENV"
echo "Template: $TEMPLATE"
echo "Model: $MODEL"
echo "Log path: $LOG_PATH"
echo "============================================"
echo ""

# Shift past the first 3 arguments to allow pass-through
shift 3 2>/dev/null || true

python -m src.train \
    env="$ENV" \
    prompt_template="$TEMPLATE" \
    model_name="$MODEL" \
    groups_per_batch=100 \
    group_size=4 \
    learning_rate=1e-5 \
    max_tokens=256 \
    eval_every=20 \
    save_every=20 \
    log_path="$LOG_PATH" \
    behavior_if_log_dir_exists=ask \
    "$@"

echo ""
echo "============================================"
echo "Training complete!"
echo "Results saved to: $LOG_PATH"
echo "============================================"
