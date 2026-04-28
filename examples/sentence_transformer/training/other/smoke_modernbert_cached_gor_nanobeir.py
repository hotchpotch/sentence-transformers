from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from datasets import load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    GlobalOrthogonalRegularizationLoss,
    GlobalOrthogonalRegularizationWrapperLoss,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "answerdotai/ModernBERT-base"
GOR_MODE = os.environ.get("GOR_MODE", "original")
OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", f"models/modernbert-gooaq-cached-1m-bs128-lr2e-5-gor0.01-{GOR_MODE}")
)
TRAIN_SAMPLES_ENV = os.environ.get("TRAIN_SAMPLES", "1000000")
TRAIN_SAMPLES: int | None = None if TRAIN_SAMPLES_ENV.lower() in {"all", "full", "none"} else int(TRAIN_SAMPLES_ENV)
MAX_STEPS_ENV = os.environ.get("MAX_STEPS")
MAX_STEPS = int(MAX_STEPS_ENV) if MAX_STEPS_ENV else None
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "128"))
CACHED_MINI_BATCH_SIZE = int(os.environ.get("CACHED_MINI_BATCH_SIZE", "32"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "128"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", "0.1"))
GOR_WEIGHT = float(os.environ.get("GOR_WEIGHT", "0.01"))
SEED = int(os.environ.get("SEED", "12"))
EVAL_PRECISIONS = ("float32", "int8", "binary")
EVAL_ATTN_IMPLEMENTATION = os.environ.get("EVAL_ATTN_IMPLEMENTATION", "eager")
RUN_LABELS = tuple(label.strip() for label in os.environ.get("RUN_LABELS", "no-gor,gor").split(",") if label.strip())
TRAIN_ONLY = os.environ.get("TRAIN_ONLY") == "1"
EVAL_ONLY = os.environ.get("EVAL_ONLY") == "1"

if GOR_MODE not in {"gemma", "original"}:
    raise ValueError(f"GOR_MODE must be 'gemma' or 'original', got {GOR_MODE!r}")


def parse_embedding_indices() -> tuple[int, ...]:
    default_indices = "0,2" if GOR_MODE == "original" else "0,1"
    raw_indices = os.environ.get("GOR_EMBEDDING_INDICES", default_indices)
    try:
        indices = tuple(int(index.strip()) for index in raw_indices.split(",") if index.strip())
    except ValueError as exc:
        raise ValueError(f"GOR_EMBEDDING_INDICES must be comma-separated integers, got {raw_indices!r}") from exc
    if not indices:
        raise ValueError("GOR_EMBEDDING_INDICES must select at least one embedding column.")
    if GOR_MODE == "original" and len(indices) < 2:
        raise ValueError("GOR_EMBEDDING_INDICES must select at least two columns for original GOR mode.")
    return indices


GOR_EMBEDDING_INDICES = parse_embedding_indices()


def build_model(run_label: str) -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
        },
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"ModernBERT base GooAQ CachedMNRL smoke run ({run_label})",
        ),
    )
    model.max_seq_length = 512
    return model


def build_loss(model: SentenceTransformer, use_gor: bool) -> torch.nn.Module:
    base_loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=CACHED_MINI_BATCH_SIZE,
    )
    if not use_gor:
        return base_loss

    if GOR_MODE == "original":
        gor_loss = GlobalOrthogonalRegularizationLoss(
            model,
            mode="original",
            embedding_indices=GOR_EMBEDDING_INDICES,
        )
    else:
        gor_loss = GlobalOrthogonalRegularizationLoss(
            model,
            mode="gemma",
            mean_weight=0.0,
            second_moment_weight=1.0,
            aggregation="sum",
            second_moment_threshold=None,
            embedding_indices=GOR_EMBEDDING_INDICES,
        )
    return GlobalOrthogonalRegularizationWrapperLoss(
        model,
        base_loss,
        gor_weight=GOR_WEIGHT,
        gor_loss=gor_loss,
    )


def train_run(train_dataset, run_label: str, use_gor: bool, max_steps: int) -> Path:
    output_dir = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}-{run_label}")
    final_dir = output_dir / "final"
    if final_dir.exists():
        logger.info("Skipping %s training; %s already exists", run_label, final_dir)
        return final_dir

    logger.info("Loading %s for %s", MODEL_NAME, run_label)
    model = build_model(run_label)
    loss = build_loss(model, use_gor=use_gor)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        logging_first_step=True,
        seed=SEED,
        data_seed=SEED,
        run_name=f"{MODEL_NAME.split('/')[-1]}-gooaq-cached-{run_label}",
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - start

    model.save(str(final_dir))
    logger.info("Saved %s model to %s", run_label, final_dir)
    timing_path = output_dir / "train-timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "train_samples": TRAIN_SAMPLES,
                "train_dataset_size": len(train_dataset),
                "max_steps": max_steps,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "cached_mini_batch_size": CACHED_MINI_BATCH_SIZE,
                "seed": SEED,
                "seconds": train_seconds,
                "minutes": train_seconds / 60,
                "metrics": train_result.metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("%s training took %.2f seconds (%.2f minutes)", run_label, train_seconds, train_seconds / 60)

    del trainer, loss, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_dir


def evaluate_nanobeir(model_dir: Path, run_label: str, precision: str) -> dict[str, float]:
    logger.info("Evaluating %s with NanoBEIR-en %s precision", run_label, precision)
    model = SentenceTransformer(
        str(model_dir),
        model_kwargs={
            "attn_implementation": EVAL_ATTN_IMPLEMENTATION,
        },
    )
    evaluator = NanoBEIREvaluator(
        dataset_id="sentence-transformers/NanoBEIR-en",
        dataset_names=None,
        precision=precision,
        batch_size=EVAL_BATCH_SIZE,
        show_progress_bar=True,
    )
    output_path = model_dir.parent / f"nanobeir-{precision}"
    start = time.perf_counter()
    results = evaluator(model, output_path=str(output_path))
    eval_seconds = time.perf_counter() - start
    logger.info("%s primary metric %s = %.6f", run_label, evaluator.primary_metric, results[evaluator.primary_metric])
    timing_path = output_path / "eval-timing.json"
    timing_path.write_text(
        json.dumps(
            {
                "run_label": run_label,
                "precision": precision,
                "seconds": eval_seconds,
                "minutes": eval_seconds / 60,
                "primary_metric": evaluator.primary_metric,
                "primary_metric_value": results[evaluator.primary_metric],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info(
        "%s %s evaluation took %.2f seconds (%.2f minutes)",
        run_label,
        precision,
        eval_seconds,
        eval_seconds / 60,
    )

    del model, evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main() -> None:
    train_dataset = None
    max_steps = MAX_STEPS
    if not EVAL_ONLY:
        if TRAIN_SAMPLES is None:
            logger.info("Loading full GooAQ train split")
            train_dataset = load_dataset("sentence-transformers/gooaq", split="train")
        else:
            logger.info("Loading GooAQ train subset: %d samples", TRAIN_SAMPLES)
            train_dataset = load_dataset("sentence-transformers/gooaq", split=f"train[:{TRAIN_SAMPLES}]")
        if max_steps is None:
            max_steps = math.ceil(len(train_dataset) / TRAIN_BATCH_SIZE)
            logger.info("Using one-epoch max_steps=%d for %d rows", max_steps, len(train_dataset))
        train_dataset = train_dataset.add_column(
            "negative",
            train_dataset["answer"][TRAIN_BATCH_SIZE:] + train_dataset["answer"][:TRAIN_BATCH_SIZE],
        )

    run_specs = {"no-gor": False, "gor": True}
    runs = {}
    for run_label, use_gor in run_specs.items():
        if run_label not in RUN_LABELS:
            continue
        if EVAL_ONLY:
            runs[run_label] = OUTPUT_ROOT.with_name(f"{OUTPUT_ROOT.name}-{run_label}") / "final"
        else:
            runs[run_label] = train_run(train_dataset, run_label, use_gor=use_gor, max_steps=max_steps)

    if TRAIN_ONLY:
        logger.info("TRAIN_ONLY=1, skipping evaluation")
        return

    all_results = {
        precision: {
            run_label: evaluate_nanobeir(model_dir, run_label, precision)
            for run_label, model_dir in runs.items()
        }
        for precision in EVAL_PRECISIONS
    }
    result_path = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}-nanobeir-results.json"
    result_path.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Wrote full results to %s", result_path)

    if {"no-gor", "gor"}.issubset(runs):
        comparison = {}
        for precision, precision_results in all_results.items():
            primary_keys = sorted(key for key in precision_results["no-gor"] if key.endswith("_ndcg@10"))
            comparison[precision] = {
                key: {
                    "no-gor": precision_results["no-gor"].get(key),
                    "gor": precision_results["gor"].get(key),
                    "delta": precision_results["gor"].get(key, 0.0) - precision_results["no-gor"].get(key, 0.0),
                }
                for key in primary_keys
            }
        comparison_path = OUTPUT_ROOT.parent / f"{OUTPUT_ROOT.name}-nanobeir-ndcg10.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("Wrote NDCG@10 comparison to %s", comparison_path)


if __name__ == "__main__":
    main()
