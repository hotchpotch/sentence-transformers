from __future__ import annotations

import numpy as np

from sentence_transformers.quantization import quantize_embeddings


def test_quantize_embeddings_int8_uses_round_not_truncation():
    embeddings = np.array([[1.6]], dtype=np.float32)
    ranges = np.array([[0.0], [255.0]], dtype=np.float32)

    quantized = quantize_embeddings(embeddings, precision="int8", ranges=ranges)

    assert quantized.dtype == np.int8
    assert quantized[0, 0] == np.int8(-126)


def test_quantize_embeddings_uint8_uses_round_not_truncation():
    embeddings = np.array([[1.6]], dtype=np.float32)
    ranges = np.array([[0.0], [255.0]], dtype=np.float32)

    quantized = quantize_embeddings(embeddings, precision="uint8", ranges=ranges)

    assert quantized.dtype == np.uint8
    assert quantized[0, 0] == np.uint8(2)


def test_quantize_embeddings_int8_and_uint8_clip_to_valid_range():
    embeddings = np.array([[300.0]], dtype=np.float32)
    ranges = np.array([[0.0], [255.0]], dtype=np.float32)

    quantized_int8 = quantize_embeddings(embeddings, precision="int8", ranges=ranges)
    quantized_uint8 = quantize_embeddings(embeddings, precision="uint8", ranges=ranges)

    assert quantized_int8[0, 0] == np.int8(127)
    assert quantized_uint8[0, 0] == np.uint8(255)
