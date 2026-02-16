from __future__ import annotations

import pytest
import torch

from sentence_transformers.losses.QuantizationAwareLoss import (
    ForwardDecorator,
    normalized_embedding_consistency_loss,
    quantize_embeddings_torch,
    resolve_warmup_steps_by_precision,
)


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


def test_resolve_warmup_steps_by_precision_defaults_and_overrides():
    warmups = resolve_warmup_steps_by_precision(
        quantization_precisions=("float32", "int8", "binary"),
        quantization_warmup_steps=200,
        quantization_warmup_steps_by_precision={"int8": 100, "binary": 800},
    )

    assert warmups["float32"] == 0
    assert warmups["int8"] == 100
    assert warmups["binary"] == 800


def test_resolve_warmup_steps_by_precision_rejects_unknown_precision():
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_warmup_steps_by_precision(
            quantization_precisions=("float32", "int8"),
            quantization_warmup_steps=200,
            quantization_warmup_steps_by_precision={"binary": 100},
        )


def test_resolve_warmup_steps_by_precision_rejects_negative_warmup():
    with pytest.raises(ValueError, match="must be >= 0"):
        resolve_warmup_steps_by_precision(
            quantization_precisions=("float32", "int8"),
            quantization_warmup_steps=200,
            quantization_warmup_steps_by_precision={"int8": -1},
        )


def test_forward_decorator_keeps_cache_float32_across_precisions():
    base_embedding = torch.tensor(
        [
            [0.0, 0.0],
            [1.1, 1.1],
            [2.0, 2.0],
        ]
    )

    def forward_fn(_features):
        return {"sentence_embedding": base_embedding.clone()}

    decorator = ForwardDecorator(forward_fn)
    decorator.start_caching()
    decorator.set_precision(None)
    _ = decorator({})
    cached_before = decorator.cache[0]["sentence_embedding"].clone()

    decorator.use_cache()
    decorator.set_precision("int8")
    _ = decorator({})

    # int8 pass must not mutate cached float32 embeddings.
    assert torch.allclose(decorator.cache[0]["sentence_embedding"], cached_before)

    decorator.use_cache()
    decorator.set_precision("binary")
    binary_output = decorator({})["sentence_embedding"]
    expected_binary = quantize_embeddings_torch(cached_before, precision="binary", binary_mode="unsigned")

    # Binary pass should quantize from original float32 cache, not int8-mutated values.
    assert torch.equal(binary_output, expected_binary)


def test_normalized_embedding_consistency_loss_is_zero_for_identical_embeddings():
    float_outputs = [{"sentence_embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}]
    quantized_outputs = [{"sentence_embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}]

    consistency = normalized_embedding_consistency_loss(quantized_outputs, float_outputs)

    assert consistency is not None
    assert torch.allclose(consistency, torch.tensor(0.0), atol=1e-7)


def test_normalized_embedding_consistency_loss_is_positive_for_shifted_embeddings():
    float_outputs = [{"sentence_embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}]
    quantized_outputs = [{"sentence_embedding": torch.tensor([[2.0, 1.0], [4.0, 3.0]])}]

    consistency = normalized_embedding_consistency_loss(quantized_outputs, float_outputs)

    assert consistency is not None
    assert consistency > 0.0
