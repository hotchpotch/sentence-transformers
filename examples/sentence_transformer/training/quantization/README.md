# Quantization-Aware Training (QAT)

Quantization-Aware Training trains embedding models to remain useful when their output embeddings are quantized to
lower-precision formats such as `int8` or `binary`.

This differs from backend/model-weight quantization. Here, the model is trained with quantized embedding outputs so that
`model.encode(..., precision="int8")` or `model.encode(..., precision="binary")` loses less quality at inference time.

## Training

Use `QuantizationAwareLoss` as a loss modifier around an existing Sentence Transformer loss:

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss, QuantizationAwareLoss

model = SentenceTransformer("microsoft/mpnet-base")
base_loss = MultipleNegativesRankingLoss(model)
loss = QuantizationAwareLoss(
    model=model,
    loss=base_loss,
    quantization_precisions=["float32", "int8", "binary"],
    quantization_weights=[1.0, 1.0, 0.5],
)
```

The loss computes the wrapped loss over each selected precision and sums the weighted results. It is compatible with
regular Sentence Transformer losses and cached in-batch-negative losses.

## Examples

- `train_qat_gooaq.py`: retrieval training on GooAQ with `MultipleNegativesRankingLoss`.
- `train_qat_nli.py`: NLI training with `MultipleNegativesRankingLoss`.
- `train_qat_sts.py`: STS Benchmark training with `CoSENTLoss`.

After training, encode or evaluate with quantized embeddings:

```python
embeddings = model.encode(sentences, precision="int8", normalize_embeddings=True)
binary_embeddings = model.encode(sentences, precision="binary", normalize_embeddings=True)
```
