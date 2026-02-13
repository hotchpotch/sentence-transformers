# QAT Experiment: final-gooaq-qat-evalonly-legacy-20260213-190857

## NDCG@10 vs Target

| Precision | Target | Pre | Post | Post-Target |
| --- | ---: | ---: | ---: | ---: |
| float32 | 0.8542 | 0.8530 | 0.8530 | -0.0012 |
| int8 | 0.8483 | 0.8475 | 0.8475 | -0.0008 |
| binary | 0.8319 | 0.8317 | 0.8317 | -0.0002 |

## Config

```json
{
  "bf16": true,
  "eval_binary_reconstruction": "zero_one",
  "eval_calibration_size": 1024,
  "eval_dequantize": true,
  "eval_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "eval_quantization_mode": "legacy",
  "fp16": false,
  "learning_rate": 2e-05,
  "num_epochs": 1.0,
  "num_eval_samples": 10000,
  "num_train_samples": 100000,
  "quantization_weights": [
    1.0,
    1.0,
    0.5
  ],
  "seed": 12,
  "skip_train": true,
  "train_batch_size": 64,
  "train_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/final-gooaq-qat-evalonly-legacy-20260213-190857`
- Final model dir: `None`
