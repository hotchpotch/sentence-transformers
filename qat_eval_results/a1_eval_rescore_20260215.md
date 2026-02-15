# A1 Result: Evaluator Oversampling + Rescore (2026-02-15)

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
| control (no rescore) | 0.5669 | 0.5669 | 0.5203 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-control-bs128-20260215-204222.json` |
| rescore x4 | 0.5669 | 0.5669 | 0.5669 | `qat_eval_results/mpnet-base-gooaq-qat-mpnet-1m-a1-evaluator-rescore4-bs128-20260215-210526.json` |

## Delta (rescore x4 - control)

- `float32`: `+0.0000`
- `int8`: `+0.0000`
- `binary`: `+0.0467`

## Observation

- Binary improved significantly under evaluator-mode rescoring.
- In this configuration, final scores became almost identical across precisions, suggesting rescore may dominate quantized ranking differences.
