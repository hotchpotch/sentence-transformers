from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import QuantizationAwareLoss
from sentence_transformers.sentence_transformer.losses.quantization_aware import quantize_embeddings_torch


class _FakeModel(torch.nn.Module):
    def forward(self, features):
        return {"sentence_embedding": features["sentence_embedding"]}


class _RecordingLoss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.seen_embeddings: list[torch.Tensor] = []

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        self.seen_embeddings.append(torch.cat([embedding.detach().clone() for embedding in embeddings]))
        return sum(embedding.pow(2).mean() for embedding in embeddings)


def test_quantize_embeddings_torch_binary_uses_straight_through_estimator() -> None:
    embeddings = torch.tensor([[-1.0, 0.0, 2.0]], requires_grad=True)

    quantized = quantize_embeddings_torch(embeddings, "binary")
    quantized.sum().backward()

    assert torch.equal(quantized.detach(), torch.tensor([[0.0, 0.0, 1.0]]))
    assert torch.equal(embeddings.grad, torch.ones_like(embeddings))


def test_quantization_aware_loss_wraps_generic_loss_without_mutating_cache() -> None:
    model = _FakeModel()
    base_loss = _RecordingLoss(model)
    loss = QuantizationAwareLoss(model, base_loss, quantization_precisions=["float32", "binary"])
    features = [
        {"sentence_embedding": torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)},
        {"sentence_embedding": torch.tensor([[0.5, -0.5], [-2.0, 1.0]], requires_grad=True)},
    ]

    output = loss(features, labels=None)

    assert set(output) == {"qat_float32", "qat_binary"}
    assert torch.equal(base_loss.seen_embeddings[0], torch.cat([feature["sentence_embedding"] for feature in features]))
    assert torch.equal(
        base_loss.seen_embeddings[1],
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    )


def test_quantization_aware_loss_validates_inputs() -> None:
    model = _FakeModel()
    base_loss = _RecordingLoss(model)

    with pytest.raises(ValueError, match="at least one quantization precision"):
        QuantizationAwareLoss(model, base_loss, quantization_precisions=[])

    with pytest.raises(ValueError, match="Invalid precision"):
        QuantizationAwareLoss(model, base_loss, quantization_precisions=["unknown"])

    with pytest.raises(ValueError, match="same length"):
        QuantizationAwareLoss(model, base_loss, quantization_precisions=["float32", "int8"], quantization_weights=[1])
