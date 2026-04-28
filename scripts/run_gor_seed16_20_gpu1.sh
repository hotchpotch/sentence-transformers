#!/usr/bin/env bash
set -euo pipefail

# Temporary one-off experiment runner used to reproduce the seed 16-20 GOR vs no-GOR queue on GPU 1.
# This intentionally keeps the exact local paths and hyperparameters from that experiment.
cd /home/hotchpotch/src/github.com/hotchpotch/sentence-transformers-wt/gor-impl

for seed in 16 17 18 19 20; do
  echo "$(date -Is) train gemma-GOR qpos seed=${seed}"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 SEED="${seed}" TRAIN_ONLY=1 RUN_LABELS=gor TRAIN_SAMPLES=full TRAIN_BATCH_SIZE=256 GOR_MODE=gemma GOR_EMBEDDING_INDICES=0,1 GOR_WEIGHT=0.005 OUTPUT_ROOT="models/modernbert-gooaq-cached-full-bs256-lr2e-5-gor0.005-gemma-qpos-seed${seed}" .venv/bin/python examples/sentence_transformer/training/other/smoke_modernbert_cached_gor_nanobeir.py

  echo "$(date -Is) eval gemma-GOR qpos seed=${seed}"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 SEED="${seed}" EVAL_ONLY=1 RUN_LABELS=gor TRAIN_SAMPLES=full TRAIN_BATCH_SIZE=256 GOR_MODE=gemma GOR_EMBEDDING_INDICES=0,1 GOR_WEIGHT=0.005 OUTPUT_ROOT="models/modernbert-gooaq-cached-full-bs256-lr2e-5-gor0.005-gemma-qpos-seed${seed}" .venv/bin/python examples/sentence_transformer/training/other/smoke_modernbert_cached_gor_nanobeir.py

  echo "$(date -Is) train no-GOR qpos seed=${seed}"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 SEED="${seed}" TRAIN_ONLY=1 RUN_LABELS=no-gor TRAIN_SAMPLES=full TRAIN_BATCH_SIZE=256 GOR_MODE=gemma GOR_EMBEDDING_INDICES=0,1 GOR_WEIGHT=0.005 OUTPUT_ROOT="models/modernbert-gooaq-cached-full-bs256-lr2e-5-gor0.005-baseline-qpos-seed${seed}" .venv/bin/python examples/sentence_transformer/training/other/smoke_modernbert_cached_gor_nanobeir.py

  echo "$(date -Is) eval no-GOR qpos seed=${seed}"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 SEED="${seed}" EVAL_ONLY=1 RUN_LABELS=no-gor TRAIN_SAMPLES=full TRAIN_BATCH_SIZE=256 GOR_MODE=gemma GOR_EMBEDDING_INDICES=0,1 GOR_WEIGHT=0.005 OUTPUT_ROOT="models/modernbert-gooaq-cached-full-bs256-lr2e-5-gor0.005-baseline-qpos-seed${seed}" .venv/bin/python examples/sentence_transformer/training/other/smoke_modernbert_cached_gor_nanobeir.py
done
