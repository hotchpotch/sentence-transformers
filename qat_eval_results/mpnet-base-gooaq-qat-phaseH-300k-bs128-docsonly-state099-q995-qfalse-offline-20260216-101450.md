# QAT Experiment: mpnet-base-gooaq-qat-phaseH-300k-bs128-docsonly-state099-q995-qfalse-offline-20260216-101450

## NDCG@10 vs Target

| Precision | Target | Pre | Post | Post-Target |
| --- | ---: | ---: | ---: | ---: |

## Config

```json
{
  "attn_fallback": true,
  "attn_implementation": null,
  "bf16": true,
  "disable_flash_attn_package": false,
  "during_train_eval_history": null,
  "eval_benchmark": "nanobeir",
  "eval_binary_reconstruction": "zero_one",
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
  "learning_rate": 2e-05,
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
  "num_eval_samples": 100,
  "num_train_samples": 300000,
  "quantization_weights": [
    1.0,
    1.0,
    0.5
  ],
  "seed": 42,
  "skip_train": false,
  "train_batch_size": 128,
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
  "train_quantization_role": "docs_only",
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": true,
  "trust_remote_code": false,
  "warmup_ratio": 0.1
}
```

- Output dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-phaseH-300k-bs128-docsonly-state099-q995-qfalse-offline-20260216-101450`
- Final model dir: `examples/sentence_transformer/training/quantization/models/mpnet-base-gooaq-qat-phaseH-300k-bs128-docsonly-state099-q995-qfalse-offline-20260216-101450/final`
