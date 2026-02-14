#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

RUNNER_SCRIPT="examples/sentence_transformer/training/quantization/run_trainb_research_rounds_configurable.sh"
CURRENT_TAG_PREFIX="${CURRENT_TAG_PREFIX:-trainb1m}"
WAIT_PID="${WAIT_PID:-}"

if [[ ! -x "${RUNNER_SCRIPT}" ]]; then
  chmod +x "${RUNNER_SCRIPT}"
fi

if [[ -n "${WAIT_PID}" ]]; then
  echo "[WAIT] Waiting for pid=${WAIT_PID} to finish..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
  echo "[WAIT] pid=${WAIT_PID} finished."
fi

echo "[WAIT] Waiting for current ${CURRENT_TAG_PREFIX} runs to finish..."
while pgrep -f "train_qat_gooaq_ablation.py.*--experiment-name ${CURRENT_TAG_PREFIX}-" >/dev/null 2>&1 || \
  pgrep -f "run_qat_with_batch_fallback.sh ${CURRENT_TAG_PREFIX}-" >/dev/null 2>&1 || \
  pgrep -f "run_trainb_1m_research_rounds.sh" >/dev/null 2>&1; do
  sleep 60
done

echo "[START] Launching mmBERT flash_attention_2 3M/1epoch research cycle"
TAG_PREFIX="mmbert3mfa2" \
MODEL_NAME="hotchpotch/mmBERT-L7H384-pruned" \
NUM_TRAIN_SAMPLES="3000000" \
NUM_EVAL_SAMPLES="10000" \
NUM_EPOCHS="1.0" \
SEED="42" \
EVAL_EVERY_TRAIN_SAMPLES="100000" \
BATCH_SIZES="128" \
ATTN_IMPLEMENTATION="flash_attention_2" \
ATTN_FALLBACK="true" \
TRUST_REMOTE_CODE="true" \
RUNS_TSV="tmp/mmbert3mfa2_research_runs.tsv" \
REPORT_MD="qat_3M_mmbert_flash2_research_report.md" \
REPORT_JSON="qat_eval_results/qat_3M_mmbert_flash2_research_report.json" \
bash "${RUNNER_SCRIPT}"

