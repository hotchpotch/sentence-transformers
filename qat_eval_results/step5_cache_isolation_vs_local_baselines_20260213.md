# Step5 Comparison (Cache Isolation Fix)

- Candidate: `mpnet-base-gooaq-qat-step5-cache-isolation-trainB-ema-warmup-20260213-213929`
- Step4b: `mpnet-base-gooaq-qat-step4b-trainB-ema-per-precision-warmup-20260213-213450`
- Baseline: `mpnet-base-gooaq-qat-local-baseline-mnrl-nanobeir-card-s42-20260213-205633`
- TrainB: `mpnet-base-gooaq-qat-local-trainB-qat-nanobeir-card-s42-20260213-205935`

- Change: ForwardDecorator cache is now copied per precision pass to avoid mutating float32 cache between quantization passes.

| Precision | Candidate | Step4b | Delta vs Step4b | Baseline | Delta vs Baseline | TrainB | Delta vs TrainB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 0.518042 | 0.520055 | -0.002013 | 0.529585 | -0.011543 | 0.520536 | -0.002494 |
| int8 | 0.507219 | 0.516528 | -0.009309 | 0.514891 | -0.007672 | 0.499400 | +0.007819 |
| binary | 0.487893 | 0.503277 | -0.015384 | 0.477290 | +0.010603 | 0.493997 | -0.006103 |

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
