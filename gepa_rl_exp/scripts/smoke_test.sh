#!/bin/bash
# Smoke test: Quick validation run (~1-2 minutes)
# Uses arithmetic environment with limited batches for speed

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
MODEL="${1:-Qwen/Qwen3-8B}"
LOG_PATH="${2:-runs/smoke_test_$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo "Math RL Smoke Test (Quick)"
echo "============================================"
echo "Model: $MODEL"
echo "Log path: $LOG_PATH"
echo "============================================"
echo ""

python -m src.train \
    env=arithmetic \
    n_batches=3 \
    model_name="$MODEL" \
    groups_per_batch=8 \
    group_size=4 \
    max_tokens=32 \
    eval_every=2 \
    save_every=2 \
    log_path="$LOG_PATH" \
    behavior_if_log_dir_exists=delete

echo ""
echo "============================================"
echo "Smoke test complete!"
echo "Results saved to: $LOG_PATH"
echo "============================================"
