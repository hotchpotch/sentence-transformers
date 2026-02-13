# QAT Experiment: mpnet-base-gooaq-qat-trainG-signed-noema-nowarmup-wb025-20260213-193328

## NDCG@10 vs Target

| Precision | Target | Pre | Post | Post-Target |
| --- | ---: | ---: | ---: | ---: |
| float32 | 0.8542 | 0.2549 | 0.8716 | +0.0174 |
| int8 | 0.8483 | 0.2466 | 0.8681 | +0.0198 |
| binary | 0.8319 | 0.4220 | 0.8552 | +0.0233 |

## Config

```json
{
  "bf16": true,
  "eval_binary_reconstruction": "minus_one_one",
  "eval_calibration_size": 1024,
  "eval_dequantize": true,
  "eval_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "eval_quantization_mode": "evaluator",
  "fp16": false,
  "learning_rate": 2e-05,
  "num_epochs": 1.0,
  "num_eval_samples": 5000,
  "num_train_samples": 40000,
  "quantization_weights": [
    1.0,
    1.0,
    0.25
  ],
  "seed": 12,
  "skip_train": false,
  "train_batch_size": 64,
  "train_binary_mode": "signed",
  "train_int8_range_momentum": 0.99,
  "train_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": false,
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-trainG-signed-noema-nowarmup-wb025-20260213-193328`
- Final model dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-trainG-signed-noema-nowarmup-wb025-20260213-193328/final`
