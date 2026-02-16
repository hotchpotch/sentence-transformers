# QAT Experiment: mpnet-base-gooaq-qat-phaseI-300k-bs1024-cachedmnrl-cachefix-lr16e4-full13-offline-20260216-105539

## NDCG@10 vs Target

| Precision | Target | Pre | Post | Post-Target |
| --- | ---: | ---: | ---: | ---: |

## Config

```json
{
  "attn_fallback": true,
  "attn_implementation": null,
  "bf16": true,
  "cached_mnrl_mini_batch_size": 64,
  "disable_flash_attn_package": false,
  "during_train_eval_history": null,
  "eval_benchmark": "nanobeir",
  "eval_binary_reconstruction": "minus_one_one",
  "eval_calibration_size": 1024,
  "eval_dequantize": true,
  "eval_during_train": false,
  "eval_every_train_samples": 0,
  "eval_int8_range_momentum": 0.99,
  "eval_int8_range_quantile": 0.995,
  "eval_int8_range_std_multiplier": 1.0,
  "eval_int8_range_strategy": "quantile",
  "eval_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "eval_quantization_mode": "evaluator",
  "eval_quantize_queries": false,
  "eval_rescore": false,
  "eval_rescore_multiplier": 2,
  "eval_samples_effective": null,
  "eval_steps_effective": null,
  "fp16": false,
  "learning_rate": 0.00016,
  "nanobeir_batch_size": 32,
  "nanobeir_dataset_id": "sentence-transformers/NanoBEIR-en",
  "nanobeir_dataset_names": [
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020"
  ],
  "num_epochs": 1.0,
  "num_eval_samples": 10000,
  "num_train_samples": 300000,
  "quantization_weights": [
    1.0,
    1.0,
    0.5
  ],
  "seed": 12,
  "skip_train": false,
  "train_base_loss": "cached_mnrl",
  "train_batch_size": 1024,
  "train_binary_mode": "signed",
  "train_consistency_weight": 0.0,
  "train_int8_range_momentum": 0.99,
  "train_loss": "qat",
  "train_precision_warmup_steps": null,
  "train_precisions": [
    "float32",
    "int8",
    "binary"
  ],
  "train_quantization_role": "all",
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": false,
  "trust_remote_code": false,
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-phaseI-300k-bs1024-cachedmnrl-cachefix-lr16e4-full13-offline-20260216-105539`
- Final model dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-phaseI-300k-bs1024-cachedmnrl-cachefix-lr16e4-full13-offline-20260216-105539/final`
