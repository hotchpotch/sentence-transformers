"""Simple dense multi-dataset Nano macro example.

Run:
  uv run --with datasets python examples/sentence_transformer/evaluation/evaluation_nano_dense_multidataset_macro.py
"""

import logging

import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

MODEL_NAME = "intfloat/multilingual-e5-small"
MULTILINGUAL_NANOBEIR_DATASET_IDS = [
    "sentence-transformers/NanoBEIR-en",
    "LiquidAI/NanoBEIR-ja",
]
CUSTOM_DATASET_IDS = [
    "hotchpotch/NanoCodeSearchNet",
]


def evaluate_dataset(model: SentenceTransformer, dataset_id: str) -> float:
    evaluator = NanoEvaluator(
        dataset_id=dataset_id,
        dataset_names=None,
        batch_size=32,
        show_progress_bar=False,
    )
    results = evaluator(model)
    if evaluator.primary_metric is None:
        raise ValueError(f"Expected evaluator.primary_metric for dataset_id={dataset_id}")
    return float(results[evaluator.primary_metric])


model = SentenceTransformer(MODEL_NAME)

multilingual_scores = [evaluate_dataset(model, dataset_id) for dataset_id in MULTILINGUAL_NANOBEIR_DATASET_IDS]
custom_scores = [evaluate_dataset(model, dataset_id) for dataset_id in CUSTOM_DATASET_IDS]

multilingual_macro = float(np.mean(multilingual_scores))
custom_macro = float(np.mean(custom_scores))
group_macro = float(np.mean([multilingual_macro, custom_macro]))

"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: intfloat/multilingual-e5-small
  Multilingual macro mean: 0.5263
  Custom macro mean: 0.7381
  Group macro mean: 0.6322
"""

print(f"Model: {MODEL_NAME}")
print(f"Multilingual macro mean: {multilingual_macro:.4f}")
print(f"Custom macro mean: {custom_macro:.4f}")
print(f"Group macro mean: {group_macro:.4f}")
