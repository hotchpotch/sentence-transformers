# QAT Experiment: final-gooaq-qat-step2-docs-only-eval-local-trainB-s42-20260213-211734

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
  "eval_during_train": true,
  "eval_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "eval_quantization_mode": "legacy",
  "eval_quantize_queries": false,
  "fp16": false,
  "learning_rate": 2e-05,
  "nanobeir_batch_size": 32,
  "nanobeir_dataset_id": "sentence-transformers/NanoBEIR-en",
  "nanobeir_dataset_names": [
    "msmarco",
    "nq"
  ],
  "num_epochs": 1.0,
  "num_eval_samples": 1000,
  "num_train_samples": 6000,
  "quantization_weights": [
    1.0,
    1.0,
    0.5
  ],
  "seed": 42,
  "skip_train": true,
  "train_batch_size": 64,
  "train_binary_mode": "unsigned",
  "train_int8_range_momentum": 0.99,
  "train_loss": "qat",
  "train_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "train_quantization_warmup_steps": 200,
  "train_use_int8_range_state": true,
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/final-gooaq-qat-step2-docs-only-eval-local-trainB-s42-20260213-211734`
- Final model dir: `None`
