"""Simple NanoEvaluator example on NanoMIRACL.

Run:
  uv run --with datasets python examples/sentence_transformer/evaluation/evaluation_nano_dense_miracl.py
"""

import logging

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoEvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

# Keep this example light: evaluate two language splits.
MODEL_NAME = "intfloat/multilingual-e5-small"
DATASET_ID = "hotchpotch/NanoMIRACL"
DATASET_SPLITS = ["en", "ja"]

model = SentenceTransformer(MODEL_NAME)
evaluator = NanoEvaluator(
    dataset_id=DATASET_ID,
    dataset_names=DATASET_SPLITS,
    batch_size=32,
    show_progress_bar=False,
)

results = evaluator(model)
"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: intfloat/multilingual-e5-small
  Dataset: hotchpotch/NanoMIRACL
  Splits: ['en', 'ja']
  Primary metric key: NanoMIRACL_mean_cosine_ndcg@10
  Primary metric value: 0.7034
"""

primary_metric = evaluator.primary_metric
if primary_metric is None:
    raise ValueError("Expected evaluator.primary_metric to be set after evaluation.")

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_ID}")
print(f"Splits: {DATASET_SPLITS}")
print(f"Primary metric key: {primary_metric}")
print(f"Primary metric value: {results[primary_metric]:.4f}")
