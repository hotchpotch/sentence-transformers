#!/bin/bash
# Sequential training script for all reranking+pruning models

echo "Starting sequential training of reranking+pruning models..."
echo "Training started at: $(date)"

# Create log directory if needed
mkdir -p ./log

# 1. Train minimal model
echo "Starting minimal dataset training..."
uv run python scripts/train_reranking_pruning_minimal.py
if [ $? -eq 0 ]; then
    echo "Minimal training completed successfully at: $(date)"
else
    echo "Minimal training failed at: $(date)"
    exit 1
fi

# 2. Train small model
echo "Starting small dataset training..."
uv run python scripts/train_reranking_pruning_small.py
if [ $? -eq 0 ]; then
    echo "Small training completed successfully at: $(date)"
else
    echo "Small training failed at: $(date)"
    exit 1
fi

# 3. Train full model
echo "Starting full dataset training..."
uv run python scripts/train_reranking_pruning_full.py
if [ $? -eq 0 ]; then
    echo "Full training completed successfully at: $(date)"
else
    echo "Full training failed at: $(date)"
    exit 1
fi

echo "All training completed successfully at: $(date)"