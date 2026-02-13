#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

RUN_FALLBACK_SCRIPT="examples/sentence_transformer/training/quantization/run_qat_with_batch_fallback.sh"
COLLECT_SCRIPT="examples/sentence_transformer/training/quantization/collect_qat_results.py"

if [[ ! -x "${RUN_FALLBACK_SCRIPT}" ]]; then
  chmod +x "${RUN_FALLBACK_SCRIPT}"
fi

mkdir -p tmp/qat_1m_logs
mkdir -p qat_eval_results

STAGE1_TSV="tmp/qat_1m_stage1_results.tsv"
STAGE2_TSV="tmp/qat_1m_stage2_results.tsv"
STAGE3_TSV="tmp/qat_1m_stage3_results.tsv"
ALL_TSV="tmp/qat_1m_all_results.tsv"

>"${STAGE1_TSV}"
>"${STAGE2_TSV}"
>"${STAGE3_TSV}"
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
      # Cache-isolation fix is now in-tree; args remain equivalent to step4b.
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    step6_binary_delay2400)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400"
      ;;
    step7_binary_weight025)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200 --quantization-weights 1.0,1.0,0.25"
      ;;
    step8_int8_momentum0995)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.995 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200"
      ;;
    step9_asym_eval_docs_only)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.99 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,1200 --no-eval-quantize-queries"
      ;;
    step10_combined)
      echo "--train-loss qat --train-binary-mode signed --train-use-int8-range-state --train-int8-range-momentum 0.995 --train-quantization-warmup-steps 0 --train-precision-warmup-steps 0,200,2400 --quantization-weights 1.0,1.0,0.25 --no-eval-quantize-queries"
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
  bash "${RUN_FALLBACK_SCRIPT}" "${tag}" "$@"
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
  local -a entry_args
  entry_args=()
  while IFS=$'\t' read -r label path group variant; do
    [[ -z "${label}" ]] && continue
    entry_args+=(--entry "${label}|${path}|${group}|${variant}")
  done <"${tsv_path}"
  uv run python "${COLLECT_SCRIPT}" \
    --title "${title}" \
    --output-md "${output_md}" \
    --output-json "${output_json}" \
    "${entry_args[@]}"
}

pick_best_stage2_variant() {
  python - "${STAGE2_TSV}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    label, path, group, variant = line.split("\t")
    if variant == "baseline_mnrl":
        continue
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    post = payload.get("post_ndcg10", {})
    float32 = post.get("float32")
    int8 = post.get("int8")
    binary = post.get("binary")
    if int8 is None or binary is None:
        continue
    score = (float(int8) + float(binary)) / 2.0
    rows.append((score, float(binary), float(int8), float(float32 or 0.0), variant))

if not rows:
    print("step4b_ema")
    raise SystemExit(0)

rows.sort(reverse=True)
print(rows[0][-1])
PY
}

append_stage1_direction() {
  python - "qat_eval_results/qat_1M_report.json" >>"qat_1M_report.md" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("rows", [])
def best(metric):
    vals = [r for r in rows if r.get(metric) is not None]
    return max(vals, key=lambda x: x[metric]) if vals else None

b_i8 = best("int8")
b_bin = best("binary")
b_ib = best("mean_int8_binary")

print("## Direction For 90k Search\n")
if b_i8:
    print(f"- Best int8 in 1M stage: `{b_i8['label']}` ({b_i8['int8']:.6f})")
if b_bin:
    print(f"- Best binary in 1M stage: `{b_bin['label']}` ({b_bin['binary']:.6f})")
if b_ib:
    print(f"- Best mean(int8,binary): `{b_ib['label']}` ({b_ib['mean_int8_binary']:.6f})")
print("- Next: run 90k search around precision warmup, binary weighting, int8 momentum, and asymmetric eval.")
PY
}

append_full_direction() {
  python - "qat_eval_results/qat_1M_report_full.json" >>"qat_1M_report_full.md" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = payload.get("rows", [])
stage3 = [r for r in rows if r.get("group") == "Stage3-1M-Final"]
if not stage3:
    raise SystemExit(0)

best = max(
    [r for r in stage3 if r.get("mean_int8_binary") is not None],
    key=lambda x: x["mean_int8_binary"],
)
baseline = next((r for r in stage3 if r.get("variant") == "baseline_mnrl"), None)

print("## Suggested Next Steps\n")
print(f"- Current best final 1M run: `{best['label']}`")
if baseline:
    for key in ["float32", "int8", "binary", "mean_int8_binary"]:
        if baseline.get(key) is not None and best.get(key) is not None:
            delta = best[key] - baseline[key]
            print(f"- Delta vs final baseline ({key}): {delta:+.6f}")
print("- Follow-up candidates: tune binary weight around 0.20-0.35 and binary warmup around 1600-3200.")
print("- Keep asymmetric eval (`quantize_queries=False`) as an eval ablation, not as the sole KPI.")
PY
}

echo "[Stage1] 1M baseline + prior step comparisons"
stage1_common=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 1000000
  --num-eval-samples 10000
  --no-eval-during-train
)

for spec in \
  "Baseline (MNRL)|1m-baseline-mnrl|baseline_mnrl" \
  "TrainB Control|1m-trainB-control|trainB_control" \
  "Step4a Warmup|1m-step4a-warmup|step4a_warmup" \
  "Step4b EMA|1m-step4b-ema|step4b_ema" \
  "Step5 CacheFix|1m-step5-cachefix|step5_cachefix"; do
  IFS='|' read -r label tag profile <<<"${spec}"
  extra_args="$(build_profile_args "${profile}")"
  # shellcheck disable=SC2206
  profile_args=( ${extra_args} )
  path="$(run_or_resume "${tag}" "${stage1_common[@]}" "${profile_args[@]}")"
  record_result "${STAGE1_TSV}" "${label}" "${path}" "Stage1-1M" "${profile}"
done

collect_from_tsv "QAT 1M Report (Stage 1)" "${STAGE1_TSV}" "qat_1M_report.md" "qat_eval_results/qat_1M_report.json"
append_stage1_direction

echo "[Stage2] 90k search around 1M winners"
stage2_common=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 100000
  --num-eval-samples 10000
  --no-eval-during-train
)

for spec in \
  "90k Baseline (MNRL)|90k-baseline-mnrl|baseline_mnrl" \
  "90k Step4b EMA|90k-step4b-ema|step4b_ema" \
  "90k Step6 Delay2400|90k-step6-delay2400|step6_binary_delay2400" \
  "90k Step7 BinWeight0.25|90k-step7-binw025|step7_binary_weight025" \
  "90k Step8 Int8Mom0.995|90k-step8-int8mom0995|step8_int8_momentum0995" \
  "90k Step9 AsymEvalDocsOnly|90k-step9-asym-docs|step9_asym_eval_docs_only" \
  "90k Step10 Combined|90k-step10-combined|step10_combined"; do
  IFS='|' read -r label tag profile <<<"${spec}"
  extra_args="$(build_profile_args "${profile}")"
  # shellcheck disable=SC2206
  profile_args=( ${extra_args} )
  path="$(run_or_resume "${tag}" "${stage2_common[@]}" "${profile_args[@]}")"
  record_result "${STAGE2_TSV}" "${label}" "${path}" "Stage2-90k" "${profile}"
done

collect_from_tsv "QAT 90k Search Report" "${STAGE2_TSV}" "qat_eval_results/qat_90k_report.md" "qat_eval_results/qat_90k_report.json"

best_variant="$(pick_best_stage2_variant)"
echo "[Stage3] Final 1M rerun with best 90k variant: ${best_variant}"

stage3_common=(
  --eval-benchmark nanobeir
  --seed 42
  --num-train-samples 1000000
  --num-eval-samples 10000
  --no-eval-during-train
)

path_baseline_final="$(run_or_resume "1m-final-baseline-mnrl" "${stage3_common[@]}" --train-loss mnrl)"
record_result "${STAGE3_TSV}" "1M Final Baseline (MNRL)" "${path_baseline_final}" "Stage3-1M-Final" "baseline_mnrl"

best_extra_args="$(build_profile_args "${best_variant}")"
# shellcheck disable=SC2206
best_profile_args=( ${best_extra_args} )
path_best_final="$(run_or_resume "1m-final-${best_variant}" "${stage3_common[@]}" "${best_profile_args[@]}")"
record_result "${STAGE3_TSV}" "1M Final Best Variant (${best_variant})" "${path_best_final}" "Stage3-1M-Final" "${best_variant}"

collect_from_tsv \
  "QAT 1M Full Report (Stage1 + Stage2 + Stage3)" \
  "${ALL_TSV}" \
  "qat_1M_report_full.md" \
  "qat_eval_results/qat_1M_report_full.json"
append_full_direction

echo "[DONE] Reports generated:"
echo "  - qat_1M_report.md"
echo "  - qat_1M_report_full.md"
