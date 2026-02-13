#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <experiment-name> [extra args for train_qat_gooaq_ablation.py]"
  echo "Example: $0 1m-baseline --train-loss mnrl --num-train-samples 1000000 --no-eval-during-train"
  exit 1
fi

experiment_name="$1"
shift

batch_sizes_csv="${BATCH_SIZES:-256,128,64}"
IFS=',' read -r -a batch_sizes <<<"$batch_sizes_csv"

mkdir -p tmp/qat_1m_logs

for bs in "${batch_sizes[@]}"; do
  log_path="tmp/qat_1m_logs/${experiment_name}-bs${bs}.log"
  run_experiment_name="${experiment_name}-bs${bs}"
  echo "[START] ${run_experiment_name} (log: ${log_path})"

  set +e
  uv run python examples/sentence_transformer/training/quantization/train_qat_gooaq_ablation.py \
    --experiment-name "${run_experiment_name}" \
    --train-batch-size "${bs}" \
    "$@" 2>&1 | tee "${log_path}"
  status=${PIPESTATUS[0]}
  set -e

  if [ "${status}" -eq 0 ]; then
    echo "[OK] ${run_experiment_name}"
    exit 0
  fi

  if rg -qi "out of memory|cuda out of memory|cublas" "${log_path}"; then
    echo "[RETRY] OOM-like failure for bs=${bs}; trying next batch size"
    continue
  fi

  echo "[FAIL] Non-OOM failure for ${run_experiment_name}"
  exit "${status}"
done

echo "[FAIL] Exhausted batch sizes (${batch_sizes_csv}) for ${experiment_name}"
exit 1
