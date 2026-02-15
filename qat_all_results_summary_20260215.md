# QAT Results Summary (mpnet, NanoBEIR, 2026-02-15)

## 対象
- 1M-bs128: Baseline / TrainB-control / Step4b / Warmup1800 / PR3655
- 3M-bs64: Baseline / TrainB-control / Step4b / Warmup1800 / PR3655-like profile
- 900k: PR3655(bs128) と PR3655-like(bs64)

注記:
- スコアはすべて `post_ndcg10`（`float32`, `int8`, `binary`）。
- `PR3655-1M/900k-bs128` は PR3655 検証用スクリプトの結果。
- `PR3655-3M/900k-bs64` は ablation スクリプトで `--train-loss qat` のみ指定した profile（run名は `mpnet-pr3655-*`）です。

## 1M-bs128
| Variant | float32 | int8 | binary | mean3 |
| --- | ---: | ---: | ---: | ---: |
| Baseline (MNRL) | 0.5723 | 0.5641 | 0.5383 | 0.5582 |
| TrainB-control | 0.5646 | 0.5731 | 0.5621 | 0.5666 |
| Step4b | 0.5664 | 0.5571 | 0.5407 | 0.5547 |
| Warmup1800 | 0.5703 | 0.5640 | 0.5457 | 0.5600 |
| PR3655 | 0.5642 | 0.5627 | 0.5533 | 0.5601 |

## 3M-bs64
| Variant | float32 | int8 | binary | mean3 |
| --- | ---: | ---: | ---: | ---: |
| Baseline (MNRL) | 0.5379 | 0.5671 | 0.5335 | 0.5462 |
| TrainB-control | 0.5502 | 0.5667 | 0.5022 | 0.5397 |
| Step4b | 0.5503 | 0.5721 | 0.5350 | 0.5525 |
| Warmup1800 | 0.5722 | 0.5636 | 0.5297 | 0.5552 |
| PR3655-like | 0.5193 | 0.5182 | 0.4757 | 0.5044 |

## 900k 比較
| Variant | float32 | int8 | binary | mean3 |
| --- | ---: | ---: | ---: | ---: |
| PR3655 (900k-bs128) | 0.5455 | 0.5628 | 0.5346 | 0.5476 |
| PR3655-like (900k-bs64) | 0.4891 | 0.5108 | 0.4883 | 0.4961 |

## Baseline 差分
`同一条件内` の Baseline 差分（Variant - Baseline）。

### 1M-bs128 基準
| Variant | Δfloat32 | Δint8 | Δbinary |
| --- | ---: | ---: | ---: |
| TrainB-control | -0.0078 | +0.0090 | +0.0239 |
| Step4b | -0.0060 | -0.0070 | +0.0025 |
| Warmup1800 | -0.0020 | -0.0001 | +0.0074 |
| PR3655 | -0.0082 | -0.0014 | +0.0151 |

### 3M-bs64 基準
| Variant | Δfloat32 | Δint8 | Δbinary |
| --- | ---: | ---: | ---: |
| TrainB-control | +0.0123 | -0.0004 | -0.0313 |
| Step4b | +0.0124 | +0.0050 | +0.0015 |
| Warmup1800 | +0.0343 | -0.0035 | -0.0038 |
| PR3655-like | -0.0186 | -0.0489 | -0.0578 |

## 1M-bs128 と 3M-bs64 の差
`3M-bs64 - 1M-bs128`（同Variant名で比較、ただし batch size が異なるため純粋なデータ量差ではない）。

| Variant | Δfloat32 | Δint8 | Δbinary |
| --- | ---: | ---: | ---: |
| Baseline | -0.0345 | +0.0030 | -0.0048 |
| TrainB-control | -0.0144 | -0.0064 | -0.0599 |
| Step4b | -0.0161 | +0.0150 | -0.0058 |
| Warmup1800 | +0.0019 | -0.0004 | -0.0160 |
| PR3655 | -0.0449 | -0.0445 | -0.0776 |

## 設定差分（主要4系統）
| Variant | train_loss | train_binary_mode | train_use_int8_range_state | train_precision_warmup_steps | train_quant_warmup_steps | quantization_weights |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | `mnrl` | `unsigned` | `true` | `-` | `200` | `[1.0,1.0,0.5]` |
| TrainB-control | `qat` | `signed` | `false(未使用)` | `-` | `0` | `[1.0,1.0,0.5]` |
| Step4b | `qat` | `signed` | `true` | `[0,200,1200]` | `0` | `[1.0,1.0,0.5]` |
| Warmup1800 | `qat` | `signed` | `true` | `[0,200,1800]` | `0` | `[1.0,1.0,0.5]` |

PR3655メモ:
- PR3655検証スクリプト側（bs128）は `QuantizationAwareLoss` に `quantization_precisions=["float32","int8","binary"]`, `quantization_weights=[1.0,1.0,0.5]` を指定。
- bs64 の `mpnet-pr3655-*` は ablation スクリプト既定値（`qat`, `unsigned`, `train_use_int8_range_state=true`, `train_quantization_warmup_steps=200`）。

## 考察（日本語）
1. 1M-bs128 では、`TrainB-control` が `int8/binary` を最も押し上げ、`float32` を少し落とす傾向が明確です。  
2. 3M-bs64 では、`Step4b` が `int8` と `binary` のバランスが最も良く、量子化耐性の観点で最有力です。  
3. `Warmup1800` は 3M-bs64 で `float32` が最良ですが、`binary` は `Step4b` より低く、目的関数を「量子化重視」に置く場合は優先度が下がります。  
4. `TrainB-control` は 3M-bs64 で `binary` が大きく悪化しており、`int8 range state` と `precision warmup` の不在が影響している可能性が高いです。  
5. PR3655系は今回の `3M-bs64` 条件で全精度悪化しました。特に `900k` の `bs128 -> bs64` で3精度すべて顕著に低下しており、単純に `bs64` へ落としたこと、あるいは profile 差（検証スクリプト vs ablation既定）の影響が大きいです。  
6. 次の有効打は、`Step4b` を基準に `3M-bs64` で `binary weight` と `precision warmup` を微調整し、`float32` を維持しながら `binary` をさらに押し上げる探索です。  

## 参照JSON
- `qat_eval_results/mpnet-base-gooaq-qat-trainb1m-r1-baseline-mnrl-bs128-20260214-092545.json`
- `qat_eval_results/mpnet-base-gooaq-qat-trainb1m-r1-trainb-control-bs128-20260214-102012.json`
- `qat_eval_results/mpnet-base-gooaq-qat-trainb1m-r1-step4b-bs128-20260214-104545.json`
- `qat_eval_results/mpnet-base-gooaq-qat-trainb1m-r1-warmup1800-bs128-20260214-111124.json`
- `qat_eval_results/mpnet-base-gooaq-qat-pr3655-nanobeir-1000000-samples-bs128-20260215-093242.json`
- `qat_eval_results/mpnet-base-gooaq-qat-pr3655-nanobeir-900000-samples-bs128-20260215-084659.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-pr3655-900k-bs64-20260215-100458.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-pr3655-3m-bs64-20260215-102436.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-baseline-mnrl-3m-bs64-20260215-112852.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-trainb-control-3m-bs64-20260215-123021.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-step4b-ref-3m-bs64-20260215-133414.json`
- `qat_eval_results/mpnet-base-gooaq-qat-mpnet-warmup1800-3m-bs64-20260215-143834.json`
