#!/bin/bash
# Evaluation script for trained checkpoints
# Uses tinker-cookbook's Inspect AI integration

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Parse arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <checkpoint_path> [model_name] [tasks]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_path  Path to tinker checkpoint (e.g., tinker://...)"
    echo "  model_name       Base model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    echo "  tasks            Inspect AI tasks to run (default: gsm8k)"
    echo ""
    echo "Example:"
    echo "  $0 tinker://my-model meta-llama/Llama-3.1-8B-Instruct gsm8k"
    exit 1
fi

CHECKPOINT_PATH="$1"
MODEL_NAME="${2:-meta-llama/Llama-3.1-8B-Instruct}"
TASKS="${3:-gsm8k}"

echo "============================================"
echo "Math RL Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Base model: $MODEL_NAME"
echo "Tasks: $TASKS"
echo "============================================"
echo ""

# Determine renderer based on model
if [[ "$MODEL_NAME" == *"Llama"* ]]; then
    RENDERER="llama3"
elif [[ "$MODEL_NAME" == *"Qwen"* ]]; then
    RENDERER="qwen3"
else
    RENDERER="role_colon"
fi

echo "Using renderer: $RENDERER"
echo ""

python -m tinker_cookbook.eval.run_inspect_evals \
    model_path="$CHECKPOINT_PATH" \
    model_name="$MODEL_NAME" \
    tasks="$TASKS" \
    renderer_name="$RENDERER"

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "============================================"
