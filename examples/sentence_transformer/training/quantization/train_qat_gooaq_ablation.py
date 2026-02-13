"""
Ablation-oriented GooAQ Quantization-Aware Training script.

This script is designed for iterative experiments. It supports:
- legacy vs evaluator-side quantized evaluation
- configurable calibration size for evaluator-side int8/uint8 quantization
- binary reconstruction mode for similarity scoring
- writing experiment artifacts to qat_eval_results/{run_name}.json/.md

Usage examples:
python train_qat_gooaq_ablation.py --experiment-name baseline-legacy
python train_qat_gooaq_ablation.py --experiment-name eval-calibrated --eval-quantization-mode evaluator
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from datasets import Dataset, load_dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator, SequentialEvaluator
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


TARGET_NDCG10 = {
    "float32": 0.8542,
    "int8": 0.8483,
    "binary": 0.8319,
}


@dataclass
class RunSummary:
    run_name: str
    model_name: str
    output_dir: str
    final_output_dir: str | None
    pre_metrics: dict[str, float]
    post_metrics: dict[str, float]
    target_ndcg10: dict[str, float]
    pre_ndcg10: dict[str, float]
    post_ndcg10: dict[str, float]
    post_delta_vs_target: dict[str, float]
    config: dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="microsoft/mpnet-base")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--train-loss", choices=["qat", "mnrl"], default="qat")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--num-train-samples", type=int, default=100_000)
    parser.add_argument("--num-eval-samples", type=int, default=10_000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-root", default="examples/sentence_transformer/training/quantization/models")
    parser.add_argument("--results-dir", default="qat_eval_results")
    parser.add_argument("--eval-benchmark", choices=["gooaq-dev", "nanobeir"], default="gooaq-dev")
    parser.add_argument("--eval-quantization-mode", choices=["legacy", "evaluator"], default="legacy")
    parser.add_argument("--eval-calibration-size", type=int, default=1024)
    parser.add_argument("--eval-int8-range-strategy", choices=["minmax", "rolling_std"], default="minmax")
    parser.add_argument("--eval-int8-range-momentum", type=float, default=0.99)
    parser.add_argument("--eval-int8-range-std-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--eval-binary-reconstruction",
        choices=["zero_one", "minus_one_one"],
        default="zero_one",
    )
    parser.add_argument("--eval-dequantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-quantize-queries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-precisions", default="float32,int8,binary")
    parser.add_argument("--nanobeir-dataset-id", default="sentence-transformers/NanoBEIR-en")
    parser.add_argument("--nanobeir-dataset-names", default="msmarco,nq")
    parser.add_argument("--nanobeir-batch-size", type=int, default=32)
    parser.add_argument("--train-precisions", default="float32,int8,binary")
    parser.add_argument("--quantization-weights", default="1.0,1.0,0.5")
    parser.add_argument("--train-binary-mode", choices=["signed", "unsigned"], default="unsigned")
    parser.add_argument("--train-use-int8-range-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-int8-range-momentum", type=float, default=0.99)
    parser.add_argument("--train-quantization-warmup-steps", type=int, default=200)
    parser.add_argument(
        "--train-precision-warmup-steps",
        default="",
        help="Optional CSV of warmup steps aligned with --train-precisions, e.g. 0,200,800",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-during-train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--eval-every-train-samples",
        type=int,
        default=0,
        help="If > 0 and --eval-during-train is enabled, run evaluation every N seen train samples (approx.).",
    )
    parser.add_argument("--save-final-model", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_csv_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def create_gooaq_evaluator(
    eval_dataset: Dataset,
    full_dataset: Dataset,
    eval_precisions: list[str],
    eval_quantization_mode: str,
    eval_calibration_size: int,
    eval_binary_reconstruction: str,
    eval_dequantize: bool,
    eval_quantize_queries: bool,
    eval_range_strategy: str,
    eval_rolling_momentum: float,
    eval_rolling_std_multiplier: float,
) -> SequentialEvaluator:
    random.seed(12)
    queries = dict(zip(eval_dataset["id"], eval_dataset["question"]))
    corpus = {qid: full_dataset[qid]["answer"] for qid in queries}
    relevant_docs = {qid: {qid} for qid in eval_dataset["id"]}

    evaluators = []
    for precision in eval_precisions:
        evaluators.append(
            InformationRetrievalEvaluator(
                corpus=corpus,
                queries=queries,
                relevant_docs=relevant_docs,
                show_progress_bar=True,
                name=f"gooaq-dev-{precision}",
                precision=precision,
                quantization_eval_mode=eval_quantization_mode,
                quantization_calibration_size=eval_calibration_size,
                quantization_range_strategy=eval_range_strategy,
                quantization_rolling_momentum=eval_rolling_momentum,
                quantization_rolling_std_multiplier=eval_rolling_std_multiplier,
                quantization_dequantize=eval_dequantize,
                binary_reconstruction=eval_binary_reconstruction,
                quantize_queries=eval_quantize_queries,
            )
        )

    return SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])


def create_nanobeir_evaluator(
    eval_precisions: list[str],
    dataset_id: str,
    dataset_names: list[str],
    batch_size: int,
    quantize_queries: bool,
    eval_quantization_mode: str,
    eval_calibration_size: int,
    eval_dequantize: bool,
    eval_binary_reconstruction: str,
    eval_range_strategy: str,
    eval_rolling_momentum: float,
    eval_rolling_std_multiplier: float,
) -> SequentialEvaluator:
    evaluators = []
    for precision in eval_precisions:
        evaluators.append(
            NanoBEIREvaluator(
                dataset_names=dataset_names,
                dataset_id=dataset_id,
                batch_size=batch_size,
                show_progress_bar=True,
                write_csv=False,
                precision=precision,
                quantize_queries=quantize_queries,
                quantization_eval_mode=eval_quantization_mode,
                quantization_calibration_size=eval_calibration_size,
                quantization_dequantize=eval_dequantize,
                binary_reconstruction=eval_binary_reconstruction,
                quantization_range_strategy=eval_range_strategy,
                quantization_rolling_momentum=eval_rolling_momentum,
                quantization_rolling_std_multiplier=eval_rolling_std_multiplier,
            )
        )
    return SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])


def extract_ndcg10(metrics: dict[str, float], eval_precisions: list[str], eval_benchmark: str) -> dict[str, float]:
    out = {}
    for precision in eval_precisions:
        candidates = []
        if eval_benchmark == "gooaq-dev":
            candidates.append(f"gooaq-dev-{precision}_cosine_ndcg@10")
        elif eval_benchmark == "nanobeir":
            candidates.extend(
                [
                    f"NanoBEIR_mean_{precision}_{precision}_cosine_ndcg@10",
                    f"NanoBEIR_mean_{precision}_cosine_ndcg@10",
                ]
            )

        value = None
        for key in candidates:
            if key in metrics:
                value = float(metrics[key])
                break

        if value is None and eval_benchmark == "nanobeir":
            pattern = re.compile(rf"^NanoBEIR_mean.*{precision}_cosine_ndcg@10$")
            for key, metric_value in metrics.items():
                if pattern.match(key):
                    value = float(metric_value)
                    break

        if value is not None:
            out[precision] = value
    return out


def sanitize_name(name: str) -> str:
    return name.replace("/", "-")


def write_results(summary: RunSummary, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{summary.run_name}.json"
    md_path = results_dir / f"{summary.run_name}.md"

    payload = asdict(summary)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# QAT Experiment: {summary.run_name}",
        "",
        "## NDCG@10 vs Target",
        "",
        "| Precision | Target | Pre | Post | Post-Target |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for precision in ["float32", "int8", "binary"]:
        target = summary.target_ndcg10.get(precision)
        pre = summary.pre_ndcg10.get(precision)
        post = summary.post_ndcg10.get(precision)
        delta = summary.post_delta_vs_target.get(precision)
        if target is None or pre is None or post is None or delta is None:
            continue
        lines.append(f"| {precision} | {target:.4f} | {pre:.4f} | {post:.4f} | {delta:+.4f} |")

    lines.extend(
        [
            "",
            "## Config",
            "",
            "```json",
            json.dumps(summary.config, indent=2, sort_keys=True),
            "```",
            "",
            f"- Output dir: `{summary.output_dir}`",
            f"- Final model dir: `{summary.final_output_dir}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_precisions = parse_csv_list(args.train_precisions, str)
    eval_precisions = parse_csv_list(args.eval_precisions, str)
    quantization_weights = parse_csv_list(args.quantization_weights, float)
    precision_warmup_steps = (
        parse_csv_list(args.train_precision_warmup_steps, int) if args.train_precision_warmup_steps.strip() else None
    )
    if len(train_precisions) != len(quantization_weights):
        raise ValueError("`--train-precisions` and `--quantization-weights` must have matching lengths.")
    quantization_warmup_steps_by_precision = None
    if precision_warmup_steps is not None:
        if len(train_precisions) != len(precision_warmup_steps):
            raise ValueError("`--train-precision-warmup-steps` must match the length of `--train-precisions`.")
        quantization_warmup_steps_by_precision = dict(zip(train_precisions, precision_warmup_steps))

    short_model_name = sanitize_name(args.model_name.split("/")[-1])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = sanitize_name(f"{short_model_name}-gooaq-qat-{args.experiment_name}-{timestamp}")
    output_dir = Path(args.output_root) / run_name
    final_output_dir = output_dir / "final"

    logger.info("Run name: %s", run_name)
    logger.info("Loading model: %s", args.model_name)
    model = SentenceTransformer(
        args.model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name=f"QAT GooAQ ablation ({args.experiment_name})",
        ),
    )

    logger.info("Loading GooAQ dataset")
    dataset = load_dataset("sentence-transformers/gooaq", split="train").select(range(args.num_train_samples))
    dataset = dataset.add_column("id", range(len(dataset)))
    dataset_dict = dataset.train_test_split(test_size=args.num_eval_samples, seed=args.seed)
    train_dataset: Dataset = dataset_dict["train"]
    eval_dataset: Dataset = dataset_dict["test"]
    logger.info("Train dataset size: %d", len(train_dataset))
    logger.info("Eval dataset size: %d", len(eval_dataset))

    if args.eval_benchmark == "gooaq-dev":
        evaluator = create_gooaq_evaluator(
            eval_dataset=eval_dataset,
            full_dataset=dataset,
            eval_precisions=eval_precisions,
            eval_quantization_mode=args.eval_quantization_mode,
            eval_calibration_size=args.eval_calibration_size,
            eval_binary_reconstruction=args.eval_binary_reconstruction,
            eval_dequantize=args.eval_dequantize,
            eval_quantize_queries=args.eval_quantize_queries,
            eval_range_strategy=args.eval_int8_range_strategy,
            eval_rolling_momentum=args.eval_int8_range_momentum,
            eval_rolling_std_multiplier=args.eval_int8_range_std_multiplier,
        )
    else:
        evaluator = create_nanobeir_evaluator(
            eval_precisions=eval_precisions,
            dataset_id=args.nanobeir_dataset_id,
            dataset_names=parse_csv_list(args.nanobeir_dataset_names, str),
            batch_size=args.nanobeir_batch_size,
            quantize_queries=args.eval_quantize_queries,
            eval_quantization_mode=args.eval_quantization_mode,
            eval_calibration_size=args.eval_calibration_size,
            eval_dequantize=args.eval_dequantize,
            eval_binary_reconstruction=args.eval_binary_reconstruction,
            eval_range_strategy=args.eval_int8_range_strategy,
            eval_rolling_momentum=args.eval_int8_range_momentum,
            eval_rolling_std_multiplier=args.eval_int8_range_std_multiplier,
        )

    logger.info("Evaluating before training")
    pre_metrics = evaluator(model)

    if not args.skip_train:
        logger.info("Starting training")
        base_loss = MultipleNegativesRankingLoss(model)
        if args.train_loss == "qat":
            loss = losses.QuantizationAwareLoss(
                model=model,
                loss=base_loss,
                quantization_precisions=train_precisions,
                quantization_weights=quantization_weights,
                binary_mode=args.train_binary_mode,
                use_int8_range_state=args.train_use_int8_range_state,
                int8_range_momentum=args.train_int8_range_momentum,
                quantization_warmup_steps=args.train_quantization_warmup_steps,
                quantization_warmup_steps_by_precision=quantization_warmup_steps_by_precision,
            )
        else:
            loss = base_loss

        eval_steps_effective: float | int | None = None
        eval_samples_effective: int | None = None
        if args.eval_during_train:
            if args.eval_every_train_samples > 0:
                eval_steps_effective = max(1, round(args.eval_every_train_samples / args.train_batch_size))
                eval_samples_effective = eval_steps_effective * args.train_batch_size
                logger.info(
                    "Eval during train: eval_steps=%d (~%d train samples between evals)",
                    eval_steps_effective,
                    eval_samples_effective,
                )
            else:
                eval_steps_effective = 0.1
                logger.info("Eval during train: eval_steps=0.1 (10%% of total train steps)")

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.train_batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            bf16=args.bf16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps" if args.eval_during_train else "no",
            eval_steps=eval_steps_effective if args.eval_during_train else None,
            save_strategy="steps" if args.eval_during_train else "no",
            save_steps=eval_steps_effective if args.eval_during_train else None,
            save_total_limit=2 if args.eval_during_train else None,
            logging_steps=0.025,
            logging_first_step=True,
            run_name=run_name,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset.remove_columns("id"),
            eval_dataset=eval_dataset.remove_columns("id") if args.eval_during_train else None,
            loss=loss,
            evaluator=evaluator if args.eval_during_train else None,
        )
        trainer.train()
        during_train_eval_history = None
        if args.eval_during_train:
            during_train_eval_history = [
                log_entry
                for log_entry in trainer.state.log_history
                if isinstance(log_entry, dict) and any(key.startswith("eval_") for key in log_entry.keys())
            ]
            logger.info("Collected %d eval points during training", len(during_train_eval_history))
        logger.info("Evaluating after training")
        post_metrics = evaluator(model)
    else:
        logger.info("Skipping training; reusing pre-training metrics as post-training metrics")
        post_metrics = pre_metrics
        eval_steps_effective = None
        eval_samples_effective = None
        during_train_eval_history = None

    if args.save_final_model and not args.skip_train:
        model.save_pretrained(str(final_output_dir))
        logger.info("Saved model to: %s", final_output_dir)

    target_ndcg10 = (
        {k: v for k, v in TARGET_NDCG10.items() if k in eval_precisions} if args.eval_benchmark == "gooaq-dev" else {}
    )
    pre_ndcg10 = extract_ndcg10(pre_metrics, eval_precisions, args.eval_benchmark)
    post_ndcg10 = extract_ndcg10(post_metrics, eval_precisions, args.eval_benchmark)
    post_delta_vs_target = {k: post_ndcg10[k] - target_ndcg10[k] for k in target_ndcg10.keys() if k in post_ndcg10}

    summary = RunSummary(
        run_name=run_name,
        model_name=args.model_name,
        output_dir=str(output_dir),
        final_output_dir=str(final_output_dir) if args.save_final_model and not args.skip_train else None,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
        target_ndcg10=target_ndcg10,
        pre_ndcg10=pre_ndcg10,
        post_ndcg10=post_ndcg10,
        post_delta_vs_target=post_delta_vs_target,
        config={
            "seed": args.seed,
            "train_loss": args.train_loss,
            "num_train_samples": args.num_train_samples,
            "num_eval_samples": args.num_eval_samples,
            "train_batch_size": args.train_batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "eval_benchmark": args.eval_benchmark,
            "eval_quantization_mode": args.eval_quantization_mode,
            "eval_calibration_size": args.eval_calibration_size,
            "eval_int8_range_strategy": args.eval_int8_range_strategy,
            "eval_int8_range_momentum": args.eval_int8_range_momentum,
            "eval_int8_range_std_multiplier": args.eval_int8_range_std_multiplier,
            "eval_binary_reconstruction": args.eval_binary_reconstruction,
            "eval_dequantize": args.eval_dequantize,
            "eval_quantize_queries": args.eval_quantize_queries,
            "nanobeir_dataset_id": args.nanobeir_dataset_id,
            "nanobeir_dataset_names": parse_csv_list(args.nanobeir_dataset_names, str),
            "nanobeir_batch_size": args.nanobeir_batch_size,
            "train_precisions": train_precisions,
            "eval_precisions": eval_precisions,
            "quantization_weights": quantization_weights,
            "train_binary_mode": args.train_binary_mode,
            "train_use_int8_range_state": args.train_use_int8_range_state,
            "train_int8_range_momentum": args.train_int8_range_momentum,
            "train_quantization_warmup_steps": args.train_quantization_warmup_steps,
            "train_precision_warmup_steps": precision_warmup_steps,
            "skip_train": args.skip_train,
            "eval_during_train": args.eval_during_train,
            "eval_every_train_samples": args.eval_every_train_samples,
            "eval_steps_effective": eval_steps_effective,
            "eval_samples_effective": eval_samples_effective,
            "during_train_eval_history": during_train_eval_history,
        },
    )
    write_results(summary, Path(args.results_dir))
    logger.info("Wrote results to: %s/%s.(json|md)", args.results_dir, run_name)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error("Run failed:\n%s", traceback.format_exc())
        raise
