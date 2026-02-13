#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

RUN_FALLBACK_SCRIPT="examples/sentence_transformer/training/quantization/run_qat_with_batch_fallback.sh"
COLLECT_SCRIPT="examples/sentence_transformer/training/quantization/collect_qat_results.py"

if [[ ! -x "${RUN_FALLBACK_SCRIPT}" ]]; then
  chmod +x "${RUN_FALLBACK_SCRIPT}"
fi

mkdir -p tmp/qat_cycle_logs
mkdir -p qat_eval_results

CYCLE_ID="${CYCLE_ID:-20260214a}"
STAGE90K_TSV="tmp/qat_cycle_${CYCLE_ID}_90k.tsv"
STAGE1M_TSV="tmp/qat_cycle_${CYCLE_ID}_1m.tsv"
ALL_TSV="tmp/qat_cycle_${CYCLE_ID}_all.tsv"

>"${STAGE90K_TSV}"
>"${STAGE1M_TSV}"
>"${ALL_TSV}"

latest_result_for_tag() {
  local tag="$1"
  ls -1t "qat_eval_results"/*-gooaq-qat-"${tag}"-bs*.json 2>/dev/null | head -n 1 || true
}

build_profile_args() {
  local profile="$1"
  case "${profile}" in
    baseline_mnrl)
      echo "--train-loss mnrl"
      ;;
    trainB_control)
      echo "--train-loss qat --train-binary-mode signed --no-train-use-int8-range-state --train-quantization-warmup-steps 0"
      ;;
    step4a_warmup)
      echo "--train-loss qat --train-binary-mode signed --no-train-use-int8-range-state --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    step4b_ema)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    step5_cachefix)
      # Cache-isolation fix is in-tree; profile args match step4b.
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    *)
      echo "[ERROR] Unknown profile: ${profile}" >&2
      exit 1
      ;;
  esac
}

run_or_resume() {
  local tag="$1"
  shift
  local existing
  existing="$(latest_result_for_tag "${tag}")"
  if [[ -n "${existing}" ]]; then
    echo "[RESUME] ${tag} -> ${existing}" >&2
    printf '%s\n' "${existing}"
    return 0
  fi

  if pgrep -f "train_qat_gooaq_ablation.py.*--experiment-name ${tag}-bs" >/dev/null 2>&1; then
    echo "[WAIT] Existing run in progress for ${tag}; waiting for completion" >&2
    while pgrep -f "train_qat_gooaq_ablation.py.*--experiment-name ${tag}-bs" >/dev/null 2>&1; do
      sleep 30
    done
    existing="$(latest_result_for_tag "${tag}")"
    if [[ -n "${existing}" ]]; then
      echo "[RESUME-AFTER-WAIT] ${tag} -> ${existing}" >&2
      printf '%s\n' "${existing}"
      return 0
    fi
  fi

  echo "[RUN] ${tag}" >&2
  bash "${RUN_FALLBACK_SCRIPT}" "${tag}" "$@" >&2
  local latest
  latest="$(latest_result_for_tag "${tag}")"
  if [[ -z "${latest}" ]]; then
    echo "[ERROR] Missing result json for tag=${tag}" >&2
    exit 1
  fi
  printf '%s\n' "${latest}"
}

record_result() {
  local tsv_path="$1"
  local label="$2"
  local path="$3"
  local group="$4"
  local variant="$5"
  printf "%s\t%s\t%s\t%s\n" "${label}" "${path}" "${group}" "${variant}" >>"${tsv_path}"
  printf "%s\t%s\t%s\t%s\n" "${label}" "${path}" "${group}" "${variant}" >>"${ALL_TSV}"
}

collect_from_tsv() {
  local title="$1"
  local tsv_path="$2"
  local output_md="$3"
  local output_json="$4"
  uv run python "${COLLECT_SCRIPT}" \
    --title "${title}" \
    --output-md "${output_md}" \
    --output-json "${output_json}" \
    --entries-file "${tsv_path}"
}

declare -a SPECS=(
  "Baseline (MNRL)|baseline_mnrl"
  "TrainB Control|trainB_control"
  "Step4a Warmup|step4a_warmup"
  "Step4b EMA|step4b_ema"
  "Step5 CacheFix|step5_cachefix"
)

echo "[Cycle90k] Running 90k cycle (no eval during train)"
cycle90k_common=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 100000
  --num-eval-samples 10000
  --no-eval-during-train
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r label profile <<<"${spec}"
  tag="90k-cycle2-${profile}"
  extra_args="$(build_profile_args "${profile}")"
  # shellcheck disable=SC2206
  profile_args=( ${extra_args} )
  path="$(run_or_resume "${tag}" "${cycle90k_common[@]}" "${profile_args[@]}")"
  record_result "${STAGE90K_TSV}" "${label}" "${path}" "Cycle90k" "${profile}"
done

collect_from_tsv \
  "QAT 90k Cycle Report (${CYCLE_ID})" \
  "${STAGE90K_TSV}" \
  "qat_eval_results/qat_90k_cycle_report.md" \
  "qat_eval_results/qat_90k_cycle_report.json"

echo "[Cycle1M] Running 1M cycle (NanoBEIR eval every 100k train samples)"
cycle1m_common=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 1000000
  --num-eval-samples 10000
  --eval-during-train
  --eval-every-train-samples 100000
)

for spec in "${SPECS[@]}"; do
  IFS='|' read -r label profile <<<"${spec}"
  tag="1m-cycle2-e100k-${profile}"
  extra_args="$(build_profile_args "${profile}")"
  # shellcheck disable=SC2206
  profile_args=( ${extra_args} )
  path="$(run_or_resume "${tag}" "${cycle1m_common[@]}" "${profile_args[@]}")"
  record_result "${STAGE1M_TSV}" "${label}" "${path}" "Cycle1M-E100kEval" "${profile}"
done

collect_from_tsv \
  "QAT 1M Cycle Report (${CYCLE_ID}, eval-every-100k-samples)" \
  "${STAGE1M_TSV}" \
  "qat_1M_report.md" \
  "qat_eval_results/qat_1M_report.json"

collect_from_tsv \
  "QAT Full Cycle Report (${CYCLE_ID}: 90k + 1M-E100kEval)" \
  "${ALL_TSV}" \
  "qat_1M_report_full.md" \
  "qat_eval_results/qat_1M_report_full.json"

echo "[DONE] Reports generated:"
echo "  - qat_eval_results/qat_90k_cycle_report.md"
echo "  - qat_1M_report.md"
echo "  - qat_1M_report_full.md"
