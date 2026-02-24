"""Simple CrossEncoder NanoBEIR reranking example.

Run:
  uv run --with datasets python examples/cross_encoder/evaluation/evaluation_nano_cross_encoder_bm25.py
"""

import logging

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

logging.basicConfig(format="%(message)s", level=logging.INFO)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
DATASET_ID = "sentence-transformers/NanoBEIR-en"
DATASET_SPLITS = ["msmarco", "nq"]

model = CrossEncoder(MODEL_NAME)
evaluator = CrossEncoderNanoBEIREvaluator(
    dataset_id=DATASET_ID,
    dataset_names=DATASET_SPLITS,
    candidate_subset_name="bm25",
    rerank_k=100,
    at_k=10,
    batch_size=32,
    show_progress_bar=False,
)

results = evaluator(model)
if evaluator.primary_metric is None:
    raise ValueError("Expected evaluator.primary_metric to be set after evaluation.")

primary_metric = evaluator.primary_metric
if primary_metric not in results:
    primary_metric = f"{evaluator.name}_{primary_metric}"
if primary_metric not in results:
    raise ValueError(f"Primary metric key not found: {primary_metric}")

"""
Example output (actual run in this repo, to be updated if defaults change):
  Model: cross-encoder/ms-marco-MiniLM-L6-v2
  Dataset: sentence-transformers/NanoBEIR-en
  Splits: ['msmarco', 'nq']
  Primary metric key: NanoBEIR_R100_mean_ndcg@10
  Primary metric value: 0.7142
"""

print(f"Model: {MODEL_NAME}")
print(f"Dataset: {DATASET_ID}")
print(f"Splits: {DATASET_SPLITS}")
print(f"Primary metric key: {primary_metric}")
print(f"Primary metric value: {float(results[primary_metric]):.4f}")
