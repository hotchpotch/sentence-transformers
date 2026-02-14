#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

RUNNER_SCRIPT="examples/sentence_transformer/training/quantization/run_trainb_research_rounds_configurable.sh"
if [[ ! -x "${RUNNER_SCRIPT}" ]]; then
  chmod +x "${RUNNER_SCRIPT}"
fi

MODEL_NAME="${MODEL_NAME:-hotchpotch/mmBERT-L7H384-pruned}"
SEED="${SEED:-42}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-1.0}"
EVAL_EVERY_TRAIN_SAMPLES="${EVAL_EVERY_TRAIN_SAMPLES:-100000}"
BATCH_SIZES="${BATCH_SIZES:-128,64,32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
ATTN_FALLBACK="${ATTN_FALLBACK:-false}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
DISABLE_FLASH_ATTN_PACKAGE="${DISABLE_FLASH_ATTN_PACKAGE:-false}"

TAG_PREFIX_90K="${TAG_PREFIX_90K:-mmbert90k}"
TAG_PREFIX_FULL="${TAG_PREFIX_FULL:-mmbertfull}"

detect_full_gooaq_samples() {
  uv run --no-sync python - <<'PY'
from datasets import load_dataset
print(len(load_dataset("sentence-transformers/gooaq", split="train")))
PY
}

FULL_SAMPLES="${FULL_SAMPLES:-}"
if [[ -z "${FULL_SAMPLES}" ]]; then
  echo "[INFO] Detecting full GooAQ sample count..."
  FULL_SAMPLES="$(detect_full_gooaq_samples)"
fi

if [[ "${FULL_SAMPLES}" -le "${NUM_EVAL_SAMPLES}" ]]; then
  echo "[ERROR] FULL_SAMPLES (${FULL_SAMPLES}) must be larger than NUM_EVAL_SAMPLES (${NUM_EVAL_SAMPLES})."
  exit 1
fi

echo "[INFO] Stage1 (90k train split from 100k total)"
TAG_PREFIX="${TAG_PREFIX_90K}" \
MODEL_NAME="${MODEL_NAME}" \
SEED="${SEED}" \
NUM_TRAIN_SAMPLES="100000" \
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
EVAL_EVERY_TRAIN_SAMPLES="${EVAL_EVERY_TRAIN_SAMPLES}" \
BATCH_SIZES="${BATCH_SIZES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
ATTN_FALLBACK="${ATTN_FALLBACK}" \
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
DISABLE_FLASH_ATTN_PACKAGE="${DISABLE_FLASH_ATTN_PACKAGE}" \
RUNS_TSV="tmp/${TAG_PREFIX_90K}_research_runs.tsv" \
REPORT_MD="qat_${TAG_PREFIX_90K}_research_report.md" \
REPORT_JSON="qat_eval_results/qat_${TAG_PREFIX_90K}_research_report.json" \
bash "${RUNNER_SCRIPT}"

echo "[INFO] Stage2 (full GooAQ 1epoch)"
echo "[INFO] FULL_SAMPLES=${FULL_SAMPLES} (train will be FULL_SAMPLES - NUM_EVAL_SAMPLES)"
TAG_PREFIX="${TAG_PREFIX_FULL}" \
MODEL_NAME="${MODEL_NAME}" \
SEED="${SEED}" \
NUM_TRAIN_SAMPLES="${FULL_SAMPLES}" \
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
EVAL_EVERY_TRAIN_SAMPLES="${EVAL_EVERY_TRAIN_SAMPLES}" \
BATCH_SIZES="${BATCH_SIZES}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
ATTN_FALLBACK="${ATTN_FALLBACK}" \
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
DISABLE_FLASH_ATTN_PACKAGE="${DISABLE_FLASH_ATTN_PACKAGE}" \
RUNS_TSV="tmp/${TAG_PREFIX_FULL}_research_runs.tsv" \
REPORT_MD="qat_${TAG_PREFIX_FULL}_research_report.md" \
REPORT_JSON="qat_eval_results/qat_${TAG_PREFIX_FULL}_research_report.json" \
bash "${RUNNER_SCRIPT}"

echo "[DONE] mmBERT 90k -> full research completed."
echo "  - 90k report: qat_${TAG_PREFIX_90K}_research_report.md"
echo "  - full report: qat_${TAG_PREFIX_FULL}_research_report.md"
