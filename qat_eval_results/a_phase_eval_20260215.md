# Phase A Eval-Side Summary (2026-02-15)

## Setup

- Dataset: `sentence-transformers/gooaq`
- Benchmark: NanoBEIR (`msmarco`, `nq`)
- Samples: train `990,000`, eval `10,000`
- Batch size: `128`
- Seed: `42`
- Train variant: `PR implementation (cache-fixed)`
- Eval mode: `evaluator`

## Results (nDCG@10)

| Variant | float32 | int8 | binary | JSON |
| --- | ---: | ---: | ---: | --- |
| A1 control (minmax, quantize_queries=true, rescore=false) | 0.5669 | 0.5669 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-control-bs128-20260215-204222.json` |
| A1 rescore x4 | 0.5669 | 0.5669 | 0.5669 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-rescore4-bs128-20260215-210526.json` |
| A2 quantile=0.995 | 0.5669 | 0.5676 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a2-evaluator-quantile0995-bs128-20260215-213110.json` |
| A3 asymmetric queries (`quantize_queries=false`) | 0.5669 | 0.5671 | 0.5508 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a3-evaluator-floatquery-bs128-20260215-215618.json` |

## Delta vs A1 Control

- A1 rescore x4:
  - `float32`: `+0.0000`
  - `int8`: `+0.0000`
  - `binary`: `+0.0467`
- A2 quantile=0.995:
  - `float32`: `+0.0000`
  - `int8`: `+0.0007`
  - `binary`: `+0.0000`
- A3 asymmetric queries:
  - `float32`: `+0.0000`
  - `int8`: `+0.0002`
  - `binary`: `+0.0306`

## Interim Takeaway

- For binary, the largest gain came from A1 rescore x4.
- A3 (asymmetric queries) also improved binary materially without enabling rescore.
- A2 quantile had a small positive effect for int8 only.
