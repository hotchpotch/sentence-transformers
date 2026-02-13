from __future__ import annotations

import torch

from sentence_transformers.losses.QuantizationAwareLoss import quantize_embeddings_torch


def test_quantize_embeddings_torch_binary_unsigned_mode():
    embeddings = torch.tensor([[-1.0, 0.0, 1.0]])
    quantized = quantize_embeddings_torch(embeddings, precision="binary", binary_mode="unsigned")

    assert torch.equal(quantized, torch.tensor([[0.0, 0.0, 1.0]]))


def test_quantize_embeddings_torch_binary_signed_mode():
    embeddings = torch.tensor([[-1.0, 0.0, 1.0]])
    quantized = quantize_embeddings_torch(embeddings, precision="binary", binary_mode="signed")

    assert torch.equal(quantized, torch.tensor([[-1.0, 1.0, 1.0]]))


def test_quantize_embeddings_torch_int8_range_state_updates_with_ema():
    range_state = {}
    embeddings_step_1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
    embeddings_step_2 = torch.tensor([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)

    quantized_step_1 = quantize_embeddings_torch(
        embeddings_step_1,
        precision="int8",
        int8_range_state=range_state,
        use_int8_range_state=True,
        int8_range_momentum=0.5,
    )
    quantized_step_2 = quantize_embeddings_torch(
        embeddings_step_2,
        precision="int8",
        int8_range_state=range_state,
        use_int8_range_state=True,
        int8_range_momentum=0.5,
    )

    assert quantized_step_1.shape == embeddings_step_1.shape
    assert quantized_step_2.shape == embeddings_step_2.shape
    assert ("int8", 2) in range_state

    mins, maxs = range_state[("int8", 2)]
    assert torch.allclose(mins, torch.tensor([[1.0, 2.0]]), atol=1e-6)
    assert torch.allclose(maxs, torch.tensor([[3.0, 4.0]]), atol=1e-6)

    quantized_step_2.sum().backward()
    assert embeddings_step_2.grad is not None
