#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

RUN_FALLBACK_SCRIPT="examples/sentence_transformer/training/quantization/run_qat_with_batch_fallback.sh"
SUMMARY_SCRIPT="examples/sentence_transformer/training/quantization/summarize_trainb_1m_research.py"

if [[ ! -x "${RUN_FALLBACK_SCRIPT}" ]]; then
  chmod +x "${RUN_FALLBACK_SCRIPT}"
fi

mkdir -p tmp/qat_1m_logs
mkdir -p qat_eval_results

RUNS_TSV="${RUNS_TSV:-tmp/trainb_1m_research_runs.tsv}"
REPORT_MD="${REPORT_MD:-qat_1M_trainb_research_report.md}"
REPORT_JSON="${REPORT_JSON:-qat_eval_results/qat_1M_trainb_research_report.json}"

>"${RUNS_TSV}"

latest_result_for_tag() {
  local tag="$1"
  ls -1t "qat_eval_results"/*-gooaq-qat-"${tag}"-bs*.json 2>/dev/null | head -n 1 || true
}

latest_baseline_json() {
  local pinned="qat_eval_results/mpnet-base-gooaq-qat-1m-cycle2-e100k-baseline_mnrl-bs128-20260214-065505.json"
  if [[ -f "${pinned}" ]]; then
    printf '%s\n' "${pinned}"
    return 0
  fi
  ls -1t qat_eval_results/*1m*baseline*mnrl*.json 2>/dev/null | head -n 1 || true
}

BASELINE_JSON="${BASELINE_JSON:-$(latest_baseline_json)}"
if [[ -z "${BASELINE_JSON}" || ! -f "${BASELINE_JSON}" ]]; then
  echo "[ERROR] Could not find baseline json. Set BASELINE_JSON=/path/to/file.json" >&2
  exit 1
fi

build_profile_args() {
  local profile="$1"
  case "${profile}" in
    baseline_mnrl)
      echo "--train-loss mnrl"
      ;;
    trainB_control)
      echo "--train-loss qat --train-binary-mode signed --no-train-use-int8-range-state --train-quantization-warmup-steps 0"
      ;;
    trainB_step4b)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    warmup_1800)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1800"
      ;;
    warmup_2400)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400"
      ;;
    warmup_3200)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,3200"
      ;;
    binw_030)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.30"
      ;;
    binw_025)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.25"
      ;;
    binw_020)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    binw_015)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.15"
      ;;
    binw_035)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.35"
      ;;
    int8mom_0995)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.995 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    int8mom_0997)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.997 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    int8mom_0980)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.98 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    qwarm_100)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 100 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    qwarm_400)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 400 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20"
      ;;
    lr_1e5)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --learning-rate 1e-5"
      ;;
    lr_15e5)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --learning-rate 1.5e-5"
      ;;
    warmup_ratio_005)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --warmup-ratio 0.05"
      ;;
    warmup_ratio_020)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --warmup-ratio 0.20"
      ;;
    asym_eval_docs_only)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --no-eval-quantize-queries"
      ;;
    combined_a)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.995 --train-quantization-warmup-steps 100 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.20 --learning-rate 1.5e-5"
      ;;
    combined_b)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.997 --train-quantization-warmup-steps 100 --train-precision-warmup-steps 0,200,3200 --quantization-weights 1.0,1.0,0.25 --learning-rate 1e-5 --warmup-ratio 0.05"
      ;;
    combined_c)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.995 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,400,2800 --quantization-weights 1.0,1.0,0.30"
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
  local round="$1"
  local label="$2"
  local tag="$3"
  local path="$4"
  printf "%s\t%s\t%s\t%s\n" "${round}" "${label}" "${tag}" "${path}" >>"${RUNS_TSV}"
}

refresh_report() {
  uv run python "${SUMMARY_SCRIPT}" \
    --tsv "${RUNS_TSV}" \
    --baseline-json "${BASELINE_JSON}" \
    --output-md "${REPORT_MD}" \
    --output-json "${REPORT_JSON}"
}

declare -a RUN_MATRIX=(
  "Round1|Baseline (MNRL)|trainb1m-r1-baseline-mnrl|baseline_mnrl"
  "Round1|TrainB Control|trainb1m-r1-trainb-control|trainB_control"
  "Round1|Step4b Ref|trainb1m-r1-step4b|trainB_step4b"
  "Round1|Warmup1800|trainb1m-r1-warmup1800|warmup_1800"
  "Round1|Warmup2400|trainb1m-r1-warmup2400|warmup_2400"
  "Round1|BinWeight0.25|trainb1m-r1-binw025|binw_025"
  "Round1|BinWeight0.20|trainb1m-r1-binw020|binw_020"
  "Round1|Int8Mom0.995|trainb1m-r1-int8mom0995|int8mom_0995"
  "Round2|Baseline (MNRL)|trainb1m-r2-baseline-mnrl|baseline_mnrl"
  "Round2|Warmup3200|trainb1m-r2-warmup3200|warmup_3200"
  "Round2|BinWeight0.15|trainb1m-r2-binw015|binw_015"
  "Round2|QWarmup100|trainb1m-r2-qwarm100|qwarm_100"
  "Round2|QWarmup400|trainb1m-r2-qwarm400|qwarm_400"
  "Round2|LR1e-5|trainb1m-r2-lr1e5|lr_1e5"
  "Round2|LR1.5e-5|trainb1m-r2-lr15e5|lr_15e5"
  "Round3|Baseline (MNRL)|trainb1m-r3-baseline-mnrl|baseline_mnrl"
  "Round3|BinWeight0.30|trainb1m-r3-binw030|binw_030"
  "Round3|BinWeight0.35|trainb1m-r3-binw035|binw_035"
  "Round3|Int8Mom0.997|trainb1m-r3-int8mom0997|int8mom_0997"
  "Round3|Int8Mom0.98|trainb1m-r3-int8mom0980|int8mom_0980"
  "Round3|WarmupRatio0.05|trainb1m-r3-warmupratio005|warmup_ratio_005"
  "Round3|WarmupRatio0.20|trainb1m-r3-warmupratio020|warmup_ratio_020"
  "Round3|AsymEvalDocsOnly|trainb1m-r3-asymdocs|asym_eval_docs_only"
  "Round3|CombinedA|trainb1m-r3-combined-a|combined_a"
  "Round3|CombinedB|trainb1m-r3-combined-b|combined_b"
  "Round3|CombinedC|trainb1m-r3-combined-c|combined_c"
)

common_args=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 1000000
  --num-eval-samples 10000
  --eval-during-train
  --eval-every-train-samples 100000
)

echo "[INFO] baseline json: ${BASELINE_JSON}"
echo "[INFO] planned runs: ${#RUN_MATRIX[@]}"

for row in "${RUN_MATRIX[@]}"; do
  IFS='|' read -r round label tag profile <<<"${row}"
  extra_args="$(build_profile_args "${profile}")"
  # shellcheck disable=SC2206
  profile_args=( ${extra_args} )
  path="$(run_or_resume "${tag}" "${common_args[@]}" "${profile_args[@]}")"
  record_result "${round}" "${label}" "${tag}" "${path}"
  refresh_report
  echo "[DONE] ${round} | ${label} -> ${path}"
done

echo "[DONE] All runs processed."
echo "  - TSV: ${RUNS_TSV}"
echo "  - Report: ${REPORT_MD}"
echo "  - Report JSON: ${REPORT_JSON}"
