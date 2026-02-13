# Step4b Comparison (EMA + Per-Precision Warmup)

- Candidate: `mpnet-base-gooaq-qat-step4b-trainB-ema-per-precision-warmup-20260213-213450`
- Step4a: `mpnet-base-gooaq-qat-step4a-trainB-per-precision-warmup-20260213-213027`
- Baseline: `mpnet-base-gooaq-qat-local-baseline-mnrl-nanobeir-card-s42-20260213-205633`
- TrainB: `mpnet-base-gooaq-qat-local-trainB-qat-nanobeir-card-s42-20260213-205935`

| Precision | Candidate | Step4a | Delta vs Step4a | Baseline | Delta vs Baseline | TrainB | Delta vs TrainB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 0.520055 | 0.515495 | +0.004559 | 0.529585 | -0.009530 | 0.520536 | -0.000481 |
| int8 | 0.516528 | 0.504440 | +0.012088 | 0.514891 | +0.001637 | 0.499400 | +0.017128 |
| binary | 0.503277 | 0.476538 | +0.026739 | 0.477290 | +0.025987 | 0.493997 | +0.009280 |

## Candidate Config

```json
{
  "seed": 42,
  "train_binary_mode": "signed",
  "train_int8_range_momentum": 0.99,
  "train_loss": "qat",
  "train_precision_warmup_steps": [
    0,
    200,
    1200
  ],
  "train_quantization_warmup_steps": 0,
  "train_use_int8_range_state": true
}
```
