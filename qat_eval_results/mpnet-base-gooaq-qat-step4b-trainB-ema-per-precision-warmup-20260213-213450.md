# QAT Experiment: mpnet-base-gooaq-qat-step4b-trainB-ema-per-precision-warmup-20260213-213450

## NDCG@10 vs Target

| Precision | Target | Pre | Post | Post-Target |
| --- | ---: | ---: | ---: | ---: |

## Config

```json
{
  "bf16": true,
  "eval_benchmark": "nanobeir",
  "eval_binary_reconstruction": "zero_one",
  "eval_calibration_size": 1024,
  "eval_dequantize": true,
  "eval_during_train": false,
  "eval_int8_range_momentum": 0.99,
  "eval_int8_range_std_multiplier": 1.0,
  "eval_int8_range_strategy": "minmax",
  "eval_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "eval_quantization_mode": "legacy",
  "eval_quantize_queries": true,
  "fp16": false,
  "learning_rate": 2e-05,
  "nanobeir_batch_size": 32,
  "nanobeir_dataset_id": "sentence-transformers/NanoBEIR-en",
  "nanobeir_dataset_names": [
    "msmarco",
    "nq"
  ],
  "num_epochs": 1.0,
  "num_eval_samples": 10000,
  "num_train_samples": 100000,
  "quantization_weights": [
    1.0,
    1.0,
    0.5
  ],
  "seed": 42,
  "skip_train": false,
  "train_batch_size": 64,
  "train_binary_mode": "signed",
  "train_int8_range_momentum": 0.99,
  "train_loss": "qat",
  "train_precision_warmup_steps": [
    0,
    200,
    1200
  ],
  "train_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": true,
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-step4b-trainB-ema-per-precision-warmup-20260213-213450`
- Final model dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-step4b-trainB-ema-per-precision-warmup-20260213-213450/final`
