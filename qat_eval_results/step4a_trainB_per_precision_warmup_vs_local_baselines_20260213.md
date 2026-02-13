# Step4a Comparison (Per-Precision Warmup)

- Candidate: `mpnet-base-gooaq-qat-step4a-trainB-per-precision-warmup-20260213-213027`
- Baseline: `mpnet-base-gooaq-qat-local-baseline-mnrl-nanobeir-card-s42-20260213-205633`
- TrainB: `mpnet-base-gooaq-qat-local-trainB-qat-nanobeir-card-s42-20260213-205935`

| Precision | Candidate | Baseline | Delta vs Baseline | TrainB | Delta vs TrainB |
| --- | ---: | ---: | ---: | ---: | ---: |
| float32 | 0.515495 | 0.529585 | -0.014089 | 0.520536 | -0.005040 |
| int8 | 0.504440 | 0.514891 | -0.010451 | 0.499400 | +0.005040 |
| binary | 0.476538 | 0.477290 | -0.000752 | 0.493997 | -0.017459 |

## Candidate Config

```json
{
  "eval_benchmark": "nanobeir",
  "eval_quantization_mode": "legacy",
  "seed": 42,
  "train_binary_mode": "signed",
  "train_loss": "qat",
  "train_precision_warmup_steps": [
    0,
    200,
    1200
  ],
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": false
}
```
