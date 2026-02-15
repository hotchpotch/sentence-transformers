# QAT Improvement Plan (2026-02-15, v2)

## Objective

Improve quantization robustness for the current Sentence-Transformers QAT workflow (train + eval), using implementation insights from Qdrant's quantization pipeline.

Primary goals:
- Maintain strong `float32` quality.
- Improve `int8` / `binary` robustness with reproducible evaluation.
- Isolate which changes actually help via controlled ablations.

## Baseline Repro (First Step)

Before introducing new changes, re-run and verify reproducibility for the 3 reference runs (`1M`, `bs=128`):

1. `MNRL baseline`
2. `PR implementation (cache-fixed)`
3. `Linear staged warmup`

Reference scores to reproduce:
- `MNRL baseline`: `0.5723 / 0.5641 / 0.5383`
- `PR implementation (cache-fixed)`: `0.5646 / 0.5731 / 0.5621`
- `Linear staged warmup`: `0.5703 / 0.5640 / 0.5457`

Re-run status (`2026-02-15`):

| Variant | Old reference (float32/int8/binary) | Re-run (float32/int8/binary) | JSON |
| --- | --- | --- | --- |
| MNRL baseline | 0.5723 / 0.5641 / 0.5383 | 0.5646 / 0.5598 / 0.5405 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-repro-baseline-mnrl-bs128-20260215-192742.json` |
| PR implementation (cache-fixed) | 0.5646 / 0.5731 / 0.5621 | 0.5669 / 0.5667 / 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-repro-trainb-control-bs128-20260215-195011.json` |
| Linear staged warmup | 0.5703 / 0.5640 / 0.5457 | 0.5658 / 0.5594 / 0.5332 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-repro-warmup1800-bs128-20260215-201313.json` |

Because reproducibility changed materially, the re-run values above are the control set for the next ablation phases.

## Qdrant-Inspired Improvement Themes

Based on Qdrant internals (`qat_report.md` + source), the most relevant ideas are:

1. **Search-stage quality control** via `oversampling + rescore`
- Qdrant uses quantized preselection + optional rescoring with higher-fidelity scoring.
- Practical implication for ST eval: evaluate quantized retrieval with explicit two-stage candidate processing, not only direct quantized similarity.

2. **Robust int8 calibration** (beyond plain min/max)
- Qdrant supports quantile-based clipping (`quantile`) to reduce outlier sensitivity.
- Practical implication for ST eval/train: add quantile-based range calibration and compare against minmax/rolling-std.

3. **Asymmetric query quantization for binary**
- Qdrant allows richer query encodings (`scalar4bits`, `scalar8bits`) against binary-indexed corpus.
- Practical implication for ST eval: test asymmetric query-side quantization for binary retrieval quality recovery.

4. **Binary encoding richness**
- Qdrant has `one_bit`, `one_and_half_bits`, `two_bits` families.
- Practical implication for ST train loss: investigate whether binary objective should include a richer/softer code target than strict 1-bit sign only.

## Execution Strategy (Sequential, Traceable)

All experiments use:
- Dataset: `sentence-transformers/gooaq`
- Effective training size: `1M request` (train `990,000` + eval `10,000`)
- Batch size: `128`
- Seed: fixed (`42`)
- Same evaluation benchmark and report format

For each change:
1. Implement exactly one improvement.
2. Run the 3 reference variants (or a minimal control pair if only eval changed).
3. Compare against current control table.
4. Log deltas, interpretation, and decision (keep/revert/iterate).

## Planned Phases

### Phase A: Eval-side improvements (no training-loss change)
- A1. Add oversampling+rescore mode to evaluator-side quantized retrieval.
- A2. Add quantile range calibration for int8 (`quantile` strategy).
- A3. Add asymmetric query encoding mode for binary eval.

Expected value: quickly identify whether quality loss is mostly from evaluation/retrieval approximation rather than train objective.

### Phase A Progress (Current)

Completed: **A1 (Evaluator oversampling + rescore)** on `1M / bs=128` with train variant `PR implementation (cache-fixed)`.

| Variant | float32 | int8 | binary | JSON |
| --- | ---: | ---: | ---: | --- |
| evaluator control (no rescore) | 0.5669 | 0.5669 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-control-bs128-20260215-204222.json` |
| evaluator rescore x4 | 0.5669 | 0.5669 | 0.5669 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-rescore4-bs128-20260215-210526.json` |

Initial interpretation:
- Binary improved strongly with rescoring (`+0.0467`).
- Scores became nearly identical across precisions in this setup, indicating rescoring may be masking quantized ranking differences.
- Next step should keep A1 as a configurable eval option, then proceed to A2 (quantile calibration) and check whether precision separation remains meaningful.

Completed: **A2 (Int8 quantile calibration)** on `1M / bs=128`, compared against A1 evaluator control (`minmax`, no rescore).

| Variant | float32 | int8 | binary | JSON |
| --- | ---: | ---: | ---: | --- |
| A1 evaluator control (minmax) | 0.5669 | 0.5669 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-control-bs128-20260215-204222.json` |
| A2 quantile (q=0.995) | 0.5669 | 0.5676 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a2-evaluator-quantile0995-bs128-20260215-213110.json` |

A2 interpretation:
- Small but positive int8 gain (`+0.0007`) versus minmax.
- No observed change for float32/binary in this setup.
- Next: proceed to A3 (asymmetric query encoding for binary) and compare against the same control.

Completed: **A3 (Asymmetric queries for quantized eval)** using `--no-eval-quantize-queries`.

| Variant | float32 | int8 | binary | JSON |
| --- | ---: | ---: | ---: | --- |
| A1 evaluator control (minmax) | 0.5669 | 0.5669 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-control-bs128-20260215-204222.json` |
| A3 asymmetric queries | 0.5669 | 0.5671 | 0.5508 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a3-evaluator-floatquery-bs128-20260215-215618.json` |

A3 interpretation:
- Binary improved materially (`+0.0306`) versus control.
- Int8 changed slightly (`+0.0002`), float32 unchanged.

Phase A summary (all A1/A2/A3 comparisons): `qat_eval_results/a_phase_eval_20260215.json` and `qat_eval_results/a_phase_eval_20260215.md`.

### Phase B: Train-loss improvements
- B1. Add ranking-consistency/distillation term between float32 and quantized similarity matrices.
- B2. Keep cache-fix and stable warmup; tune quantized loss weighting based on A-phase findings.
- B3. Test warmup defined by ratio as well as fixed steps (to avoid short-run under-warmup issues).

Expected value: improve transfer from float training geometry to quantized ranking behavior.

### Phase C: Joint tuning
- Combine best A-phase eval setup with best B-phase train setup.
- Re-run full comparison table at `1M, bs=128`.
- Optionally validate on longer-run setting once stable.

## Reporting Rules

- Every run must produce both `.json` and `.md` artifacts in `qat_eval_results/`.
- Maintain a cumulative experiment table with:
  - variant name
  - config diff
  - float32/int8/binary nDCG@10
  - delta vs control
  - decision
- If a change does not help, keep it documented and revert from the candidate default path.

## Current Working Assumption

The current instability is likely not from a single factor. It appears to be a combination of:
- quantization calibration sensitivity,
- retrieval-stage approximation effects,
- and train-objective mismatch between float and quantized ranking behavior.

This plan is designed to separate these factors and identify a robust recipe.
