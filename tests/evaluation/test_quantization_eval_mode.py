from __future__ import annotations

import numpy as np
import torch

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, InformationRetrievalEvaluator
from sentence_transformers.quantization import quantize_embeddings


def _build_ir_evaluator(**kwargs) -> InformationRetrievalEvaluator:
    queries = {"q1": "query"}
    corpus = {"d1": "doc"}
    relevant_docs = {"q1": {"d1"}}
    return InformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=relevant_docs, **kwargs)


class _RecordingModel:
    def __init__(self) -> None:
        self.calls = []
        self.similarity_fn_name = "cosine"

    def similarity(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.matmul(query_embeddings, corpus_embeddings.T)

    def encode_query(self, sentences, **kwargs) -> torch.Tensor:
        self.calls.append(("query", kwargs.get("precision")))
        return torch.zeros((len(sentences), 2), dtype=torch.float32)

    def encode_document(self, sentences, **kwargs) -> torch.Tensor:
        self.calls.append(("document", kwargs.get("precision")))
        return torch.zeros((len(sentences), 2), dtype=torch.float32)


def test_ir_evaluator_int8_dequantization():
    evaluator = _build_ir_evaluator(precision="int8", quantization_dequantize=True)
    ranges = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    quantized = torch.tensor([[0, -128]], dtype=torch.int8)

    converted = evaluator._convert_to_float(quantized, ranges=ranges)
    expected = torch.tensor([[0.00392157, 0.0]], dtype=torch.float32)

    assert torch.allclose(converted, expected, atol=1e-6)


def test_ir_evaluator_binary_minus_one_one_reconstruction():
    evaluator = _build_ir_evaluator(precision="binary", binary_reconstruction="minus_one_one")
    float_embeddings = np.array([[0.1, -0.1, 2.0, -3.0, 0.0, 1.0, -2.0, 0.5]], dtype=np.float32)
    quantized = quantize_embeddings(float_embeddings, precision="binary")

    converted = evaluator._convert_to_float(torch.from_numpy(quantized))
    expected = torch.tensor([[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]], dtype=torch.float32)

    assert torch.equal(converted, expected)


def test_embedding_similarity_binary_minus_one_one_reconstruction():
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=["a"],
        sentences2=["b"],
        scores=[1.0],
        precision="binary",
        binary_reconstruction="minus_one_one",
    )
    float_embeddings = np.array([[0.1, -0.1, 2.0, -3.0, 0.0, 1.0, -2.0, 0.5]], dtype=np.float32)
    quantized = quantize_embeddings(float_embeddings, precision="binary")

    converted = evaluator._convert_to_float(quantized)
    expected = np.array([[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0]], dtype=np.float32)

    assert np.array_equal(converted, expected)


def test_embedding_similarity_int8_dequantization():
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=["a"],
        sentences2=["b"],
        scores=[1.0],
        precision="int8",
        quantization_dequantize=True,
    )
    ranges = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    quantized = np.array([[0, -128]], dtype=np.int8)

    converted = evaluator._convert_to_float(quantized, ranges=ranges)
    expected = np.array([[0.00392157, 0.0]], dtype=np.float32)

    assert np.allclose(converted, expected, atol=1e-6)


def test_ir_evaluator_docs_only_quantization_keeps_query_float32():
    evaluator = _build_ir_evaluator(precision="int8", quantize_queries=False, quantization_eval_mode="legacy")
    model = _RecordingModel()

    evaluator.embed_inputs(model, evaluator.queries, encode_fn_name="query")
    evaluator.embed_inputs(model, evaluator.corpus, encode_fn_name="document")

    assert model.calls[0] == ("query", "float32")
    assert model.calls[1] == ("document", "int8")
