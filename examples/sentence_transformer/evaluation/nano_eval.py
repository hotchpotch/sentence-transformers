from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator, NanoEvaluator, SequentialEvaluator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SentenceTransformer on NanoBEIR + custom Nano dataset")
    parser.add_argument("--model", required=True, help="Model name or local path")

    parser.add_argument("--nanobeir-dataset-id", default="sentence-transformers/NanoBEIR-en")
    parser.add_argument("--nanobeir-datasets", default="msmarco,nq", help="Comma-separated NanoBEIR dataset names")

    parser.add_argument("--extra-dataset-id", default="hotchpotch/NanoCodeSearchNet")
    parser.add_argument(
        "--extra-splits",
        default=None,
        help="Comma-separated split names for the extra dataset. If omitted, all query splits are used.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--query-prompt", default=None)
    parser.add_argument("--corpus-prompt", default=None)
    return parser.parse_args()


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def build_evaluators(args: argparse.Namespace) -> list[Any]:
    nanobeir_dataset_names = parse_csv(args.nanobeir_datasets)
    extra_splits = parse_csv(args.extra_splits)

    nanobeir_evaluator = NanoBEIREvaluator(
        dataset_names=nanobeir_dataset_names,
        dataset_id=args.nanobeir_dataset_id,
        batch_size=args.batch_size,
        show_progress_bar=args.show_progress,
        write_csv=False,
        query_prompts=args.query_prompt,
        corpus_prompts=args.corpus_prompt,
    )

    extra_evaluator = NanoEvaluator(
        dataset_names=extra_splits,
        dataset_id=args.extra_dataset_id,
        batch_size=args.batch_size,
        show_progress_bar=args.show_progress,
        write_csv=False,
        query_prompts=args.query_prompt,
        corpus_prompts=args.corpus_prompt,
    )

    return [nanobeir_evaluator, extra_evaluator]


def main() -> None:
    args = parse_args()
    model = SentenceTransformer(args.model)

    evaluators = build_evaluators(args)
    sequential = SequentialEvaluator(
        evaluators,
        main_score_function=lambda scores: float(np.mean(scores)),
    )

    metrics = sequential(model)
    ndcg_metrics = {key: value for key, value in metrics.items() if "ndcg@10" in key}
    if not ndcg_metrics:
        logger.warning("No ndcg@10 metrics were found in the evaluation output.")

    print(
        json.dumps(
            {"primary_metric": sequential.primary_metric, "metrics": ndcg_metrics}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
