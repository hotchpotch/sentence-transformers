"""Simple sparse multi-dataset Nano macro example.

Run:
  uv run --with datasets python examples/sparse_encoder/evaluation/sparse_nano_multidataset_macro_evaluator.py
"""

import logging

import numpy as np

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

MODEL_NAME = "sparse-encoder/example-inference-free-splade-distilbert-base-uncased-nq"
MULTILINGUAL_NANOBEIR_DATASET_IDS = [
    "sentence-transformers/NanoBEIR-en",
]
CUSTOM_DATASET_IDS = [
    "hotchpotch/NanoCodeSearchNet",
]


def evaluate_dataset(model: SparseEncoder, dataset_id: str) -> float:
    evaluator = SparseNanoEvaluator(
        dataset_id=dataset_id,
        dataset_names=None,
        batch_size=32,
        show_progress_bar=False,
    )
    results = evaluator(model)
    if evaluator.primary_metric is None:
        raise ValueError(f"Expected evaluator.primary_metric for dataset_id={dataset_id}")
    return float(results[evaluator.primary_metric])


model = SparseEncoder(MODEL_NAME)

multilingual_scores = [evaluate_dataset(model, dataset_id) for dataset_id in MULTILINGUAL_NANOBEIR_DATASET_IDS]
custom_scores = [evaluate_dataset(model, dataset_id) for dataset_id in CUSTOM_DATASET_IDS]

multilingual_macro = float(np.mean(multilingual_scores))
custom_macro = float(np.mean(custom_scores))
group_macro = float(np.mean([multilingual_macro, custom_macro]))

"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: sparse-encoder/example-inference-free-splade-distilbert-base-uncased-nq
  Multilingual macro mean: 0.5205
  Custom macro mean: 0.5867
  Group macro mean: 0.5536
"""

print(f"Model: {MODEL_NAME}")
print(f"Multilingual macro mean: {multilingual_macro:.4f}")
print(f"Custom macro mean: {custom_macro:.4f}")
print(f"Group macro mean: {group_macro:.4f}")
