from __future__ import annotations

"""Run NanoBEIR + optional custom Nano evaluation in one command.

Tips:
- When datasets/models are already cached locally and you run large evaluations,
  setting `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` can reduce Hugging Face API
  calls and improve startup/evaluation stability.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator, NanoEvaluator, SentenceEvaluator, SequentialEvaluator
from sentence_transformers.model_card import get_versions

logger = logging.getLogger(__name__)

COLLECTION_DATASET_IDS: dict[str, list[str]] = {
    "nanobeir": [
        "sentence-transformers/NanoBEIR-en",
    ],
    "mnanobeir": [
        "sentence-transformers/NanoBEIR-en",
        "lightonai/NanoBEIR-ar",
        "lightonai/NanoBEIR-de",
        "lightonai/NanoBEIR-es",
        "lightonai/NanoBEIR-it",
        "lightonai/NanoBEIR-fr",
        "lightonai/NanoBEIR-pt",
        "lightonai/NanoBEIR-no",
        "lightonai/NanoBEIR-sv",
        "Serbian-AI-Society/NanoBEIR-sr",
        "LiquidAI/NanoBEIR-ko",
        "LiquidAI/NanoBEIR-ja",
        "sionic-ai/NanoBEIR-vi",
        "sionic-ai/NanoBEIR-th",
    ],
}

COLLECTION_ALIASES: dict[str, str] = {
    "nanobeir": "nanobeir",
    "mnanobeir": "mnanobeir",
    "multilingualnanobeir": "mnanobeir",
}

NANOBEIR_DATASET_ID_TO_HUMAN_READABLE_PREFIX: dict[str, str] = {
    "sentence-transformers/NanoBEIR-en": "MNanoBEIR_en",
    "lightonai/NanoBEIR-ar": "MNanoBEIR_ar",
    "lightonai/NanoBEIR-de": "MNanoBEIR_de",
    "lightonai/NanoBEIR-es": "MNanoBEIR_es",
    "lightonai/NanoBEIR-it": "MNanoBEIR_it",
    "lightonai/NanoBEIR-fr": "MNanoBEIR_fr",
    "lightonai/NanoBEIR-pt": "MNanoBEIR_pt",
    "lightonai/NanoBEIR-no": "MNanoBEIR_no",
    "lightonai/NanoBEIR-sv": "MNanoBEIR_sv",
    "Serbian-AI-Society/NanoBEIR-sr": "MNanoBEIR_sr",
    "LiquidAI/NanoBEIR-ko": "MNanoBEIR_ko",
    "LiquidAI/NanoBEIR-ja": "MNanoBEIR_ja",
    "sionic-ai/NanoBEIR-vi": "MNanoBEIR_vi",
    "sionic-ai/NanoBEIR-th": "MNanoBEIR_th",
    "hotchpotch/NanoMIRACL": "NanoMIRACL",
    "hotchpotch/NanoCodeSearchNet": "NanoCodeSearchNet",
}

COLLECTION_TO_EVALUATOR_KIND: dict[str, str] = {
    "nanobeir": "nanobeir",
    "mnanobeir": "nanobeir",
}


class PrefixedEvaluator(SentenceEvaluator):
    """Prefix evaluator metrics to avoid key collisions in SequentialEvaluator."""

    def __init__(self, evaluator: SentenceEvaluator, prefix: str) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.prefix = prefix

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        metrics = self.evaluator(model, output_path=output_path, epoch=epoch, steps=steps)
        if not isinstance(metrics, dict):
            raise ValueError("PrefixedEvaluator requires an evaluator that returns dict metrics.")

        prefixed_metrics = {f"{self.prefix}_{key}": value for key, value in metrics.items()}

        base_primary_metric = getattr(self.evaluator, "primary_metric", None)
        if base_primary_metric and base_primary_metric in metrics:
            self.primary_metric = f"{self.prefix}_{base_primary_metric}"
        elif prefixed_metrics:
            self.primary_metric = next(iter(prefixed_metrics.keys()))

        return prefixed_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SentenceTransformer on NanoBEIR collections and an optional custom Nano dataset"
    )
    parser.add_argument("--model", required=True, help="Model name or local path")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model load dtype. Defaults to bf16.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation (e.g. 'flash_attention_2').",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        help="Shortcut for '--attn-implementation flash_attention_2'.",
    )
    parser.add_argument("--device", default=None, help="Optional device override (e.g. 'cuda', 'cpu').")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for model loading.")

    parser.add_argument(
        "-c",
        "--collection",
        action="append",
        default=None,
        help=(
            "Named NanoBEIR collection(s), repeatable or comma-separated. "
            "Supported: NanoBEIR, MNanoBEIR, MultilingualNanoBEIR"
        ),
    )
    parser.add_argument(
        "--nanobeir-dataset-id",
        default=None,
        help=(
            "Optional extra NanoBEIR dataset id(s), repeatable via comma. "
            "Used in addition to --collection. Example: sentence-transformers/NanoBEIR-en"
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset-id",
        action="append",
        default=None,
        help=(
            "Optional generic NanoEvaluator dataset id(s), repeatable or comma-separated. "
            "Useful when you want to evaluate datasets directly without --collection "
            "(e.g. -d hotchpotch/NanoCodeSearchNet,hotchpotch/NanoMIRACL)."
        ),
    )
    parser.add_argument(
        "-s",
        "--split-target",
        action="append",
        default=None,
        help=(
            "NanoBEIR split target(s) like msmarco,nq. Repeatable and comma-separated. "
            "Applies to all selected collections."
        ),
    )
    parser.add_argument(
        "--nanobeir-datasets",
        default=None,
        help="Deprecated alias for split targets when --split-target is not set.",
    )

    parser.add_argument(
        "--extra-dataset-id",
        default=None,
        help="Optional extra Nano-style dataset id for generic NanoEvaluator (e.g. hotchpotch/NanoCodeSearchNet).",
    )
    parser.add_argument(
        "--extra-splits",
        default=None,
        help="Comma-separated split names for --extra-dataset-id. If omitted, all query splits are used.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--query-prompt", default=None)
    parser.add_argument("--corpus-prompt", default=None)
    parser.add_argument(
        "--output-dir",
        default="output",
        help=(
            "Base output directory. Evaluation files are saved to "
            "{output-dir}/{model_name}-{collection_or_dataset_suffix}/."
        ),
    )
    parser.add_argument(
        "--no-evaluator-csv",
        action="store_true",
        help="Disable sentence-transformers evaluator CSV outputs.",
    )
    return parser.parse_args()


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_repeated_csv(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []

    parsed: list[str] = []
    for value in values:
        parsed.extend(parse_csv(value))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def resolve_collections(collection_values: Iterable[str] | None) -> list[str]:
    raw_values = parse_repeated_csv(collection_values)
    resolved: list[str] = []
    for raw in raw_values:
        normalized = COLLECTION_ALIASES.get(raw.lower())
        if normalized is None:
            raise ValueError(f"Unknown collection '{raw}'. Supported: NanoBEIR, MNanoBEIR, MultilingualNanoBEIR")
        resolved.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for value in resolved:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def resolve_collection_dataset_entries(args: argparse.Namespace) -> list[tuple[str, str]]:
    collection_names = resolve_collections(args.collection)

    entries: list[tuple[str, str]] = []
    for collection_name in collection_names:
        evaluator_kind = COLLECTION_TO_EVALUATOR_KIND[collection_name]
        for dataset_id in COLLECTION_DATASET_IDS[collection_name]:
            entries.append((evaluator_kind, dataset_id))

    if args.nanobeir_dataset_id is not None:
        for dataset_id in parse_csv(args.nanobeir_dataset_id):
            entries.append(("nanobeir", dataset_id))
    if args.dataset_id is not None:
        for dataset_id in parse_repeated_csv(args.dataset_id):
            entries.append(("nano", dataset_id))

    if not entries:
        entries = [("nanobeir", "sentence-transformers/NanoBEIR-en")]

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if entry in seen:
            continue
        seen.add(entry)
        deduped.append(entry)
    return deduped


def resolve_split_targets(args: argparse.Namespace, collection_entries: list[tuple[str, str]]) -> list[str] | None:
    has_nanobeir = any(kind == "nanobeir" for kind, _ in collection_entries)
    has_non_nanobeir = any(kind != "nanobeir" for kind, _ in collection_entries)

    split_targets = parse_repeated_csv(args.split_target)
    if split_targets:
        return split_targets

    fallback_targets = parse_csv(args.nanobeir_datasets)
    if fallback_targets:
        return fallback_targets

    # Keep historical default split-targets for NanoBEIR-only runs.
    if has_nanobeir and not has_non_nanobeir:
        return ["msmarco", "nq"]
    return None


def dataset_id_to_prefix(dataset_id: str) -> str:
    if dataset_id in NANOBEIR_DATASET_ID_TO_HUMAN_READABLE_PREFIX:
        return NANOBEIR_DATASET_ID_TO_HUMAN_READABLE_PREFIX[dataset_id]
    return dataset_id.replace("/", "__").replace("-", "_")


def build_evaluators(args: argparse.Namespace) -> list[SentenceEvaluator]:
    collection_entries = resolve_collection_dataset_entries(args)
    split_targets = resolve_split_targets(args, collection_entries)
    extra_splits = parse_csv(args.extra_splits)
    write_evaluator_csv = not args.no_evaluator_csv

    evaluators: list[SentenceEvaluator] = []
    for evaluator_kind, dataset_id in collection_entries:
        if evaluator_kind == "nanobeir":
            evaluator = NanoBEIREvaluator(
                dataset_names=split_targets,
                dataset_id=dataset_id,
                batch_size=args.batch_size,
                show_progress_bar=args.show_progress,
                write_csv=write_evaluator_csv,
                query_prompts=args.query_prompt,
                corpus_prompts=args.corpus_prompt,
            )
        else:
            evaluator = NanoEvaluator(
                dataset_names=split_targets or None,
                dataset_id=dataset_id,
                batch_size=args.batch_size,
                show_progress_bar=args.show_progress,
                write_csv=write_evaluator_csv,
                query_prompts=args.query_prompt,
                corpus_prompts=args.corpus_prompt,
            )
        evaluators.append(PrefixedEvaluator(evaluator, prefix=dataset_id_to_prefix(dataset_id)))

    if args.extra_dataset_id is not None:
        extra_evaluator = NanoEvaluator(
            dataset_names=extra_splits or None,
            dataset_id=args.extra_dataset_id,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress,
            write_csv=write_evaluator_csv,
            query_prompts=args.query_prompt,
            corpus_prompts=args.corpus_prompt,
        )
        evaluators.append(extra_evaluator)

    return evaluators


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _resolve_attn_implementation(args: argparse.Namespace) -> str | None:
    attn_implementation: str | None = args.attn_implementation
    if args.flash_attn2:
        if attn_implementation and attn_implementation != "flash_attention_2":
            raise ValueError("Both --flash-attn2 and --attn-implementation were provided with conflicting values.")
        attn_implementation = "flash_attention_2"
    return attn_implementation


def load_model(args: argparse.Namespace) -> SentenceTransformer:
    attn_implementation = _resolve_attn_implementation(args)

    model_kwargs: dict[str, Any] = {"dtype": _resolve_dtype(args.dtype)}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    return SentenceTransformer(
        args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        model_kwargs=model_kwargs,
    )


def _resolve_effective_query_prompt(
    model: SentenceTransformer, cli_query_prompt: str | None
) -> tuple[str | None, str]:
    if cli_query_prompt is not None:
        return cli_query_prompt, "cli(--query-prompt)"

    model_prompts = model.prompts or {}
    query_prompt = model_prompts.get("query")
    if query_prompt not in (None, ""):
        return query_prompt, "model.prompts['query']"

    if model.default_prompt_name:
        default_prompt = model_prompts.get(model.default_prompt_name)
        if default_prompt not in (None, ""):
            return default_prompt, f"model.default_prompt_name='{model.default_prompt_name}'"

    return None, "none"


def _resolve_effective_corpus_prompt(
    model: SentenceTransformer, cli_corpus_prompt: str | None
) -> tuple[str | None, str]:
    if cli_corpus_prompt is not None:
        return cli_corpus_prompt, "cli(--corpus-prompt)"

    model_prompts = model.prompts or {}
    for prompt_name in ["document", "passage", "corpus"]:
        prompt = model_prompts.get(prompt_name)
        if prompt not in (None, ""):
            return prompt, f"model.prompts['{prompt_name}']"

    if model.default_prompt_name:
        default_prompt = model_prompts.get(model.default_prompt_name)
        if default_prompt not in (None, ""):
            return default_prompt, f"model.default_prompt_name='{model.default_prompt_name}'"

    return None, "none"


def _resolve_effective_prompts(model: SentenceTransformer, args: argparse.Namespace) -> dict[str, str | None]:
    query_prompt, query_source = _resolve_effective_query_prompt(model, args.query_prompt)
    corpus_prompt, corpus_source = _resolve_effective_corpus_prompt(model, args.corpus_prompt)

    return {
        "query_prompt": query_prompt,
        "query_prompt_source": query_source,
        "corpus_prompt": corpus_prompt,
        "corpus_prompt_source": corpus_source,
    }


def _print_effective_prompts(prompt_info: dict[str, str | None]) -> None:
    query_prompt = prompt_info["query_prompt"]
    query_source = prompt_info["query_prompt_source"]
    corpus_prompt = prompt_info["corpus_prompt"]
    corpus_source = prompt_info["corpus_prompt_source"]

    if query_prompt is not None:
        print(f"query prompt [{query_source}]: {query_prompt!r}")
    else:
        print("query prompt [none]: None")

    if corpus_prompt is not None:
        print(f"doc prompt [{corpus_source}]: {corpus_prompt!r}")
    else:
        print("doc prompt [none]: None")


def _collect_eval_plan(args: argparse.Namespace) -> dict[str, Any]:
    collection_entries = resolve_collection_dataset_entries(args)
    split_targets = resolve_split_targets(args, collection_entries)
    plan: dict[str, Any] = {
        "collection_entries": [
            {"evaluator_kind": kind, "dataset_id": dataset_id} for kind, dataset_id in collection_entries
        ],
        "split_targets": split_targets,
    }
    if args.extra_dataset_id is not None:
        plan["extra_dataset_id"] = args.extra_dataset_id
        plan["extra_splits"] = parse_csv(args.extra_splits) or None
    return plan


def _print_eval_plan(plan: dict[str, Any]) -> None:
    dataset_ids = [entry["dataset_id"] for entry in plan["collection_entries"]]
    print(f"collection dataset ids: {dataset_ids}")
    split_targets = plan["split_targets"]
    print(f"split targets: {split_targets if split_targets is not None else 'all'}")
    extra_dataset_id = plan.get("extra_dataset_id")
    if extra_dataset_id is not None:
        extra_splits = plan.get("extra_splits")
        print(f"extra dataset id: {extra_dataset_id}")
        print(f"extra splits: {extra_splits if extra_splits else 'all'}")


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _resolve_model_output_name(model_name_or_path: str) -> str:
    normalized = model_name_or_path.strip().rstrip("/\\")
    if not normalized:
        return "model"
    model_name = normalized.replace("\\", "/").split("/")[-1]
    safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in model_name)
    return safe_name or "model"


def _sanitize_output_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _resolve_target_output_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []

    collection_names = resolve_collections(args.collection)
    if collection_names:
        parts.extend(collection_names)

    collection_entries = resolve_collection_dataset_entries(args)
    split_targets = resolve_split_targets(args, collection_entries)
    if split_targets:
        parts.extend(split_targets)

    explicit_dataset_ids: list[str] = []
    if args.nanobeir_dataset_id is not None:
        explicit_dataset_ids.extend(parse_csv(args.nanobeir_dataset_id))
    if args.dataset_id is not None:
        explicit_dataset_ids.extend(parse_repeated_csv(args.dataset_id))
    if args.extra_dataset_id is not None:
        explicit_dataset_ids.append(args.extra_dataset_id)
    if explicit_dataset_ids:
        parts.extend(dataset_id_to_prefix(dataset_id) for dataset_id in explicit_dataset_ids)

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)

    if not deduped:
        return "default"
    return "_".join(_sanitize_output_part(part) for part in deduped)


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    model_name = _resolve_model_output_name(args.model)
    target_suffix = _resolve_target_output_suffix(args)
    output_dir = Path(args.output_dir) / f"{model_name}-{target_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_eval_json(output_dir: Path, payload: dict[str, Any]) -> Path:
    json_path = output_dir / "eval.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def _write_eval_csv(output_dir: Path, metrics: dict[str, float]) -> Path:
    csv_path = output_dir / "eval.csv"
    lines = ["metric,value"]
    for key in sorted(metrics):
        lines.append(f"{key},{metrics[key]}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path


def _write_eval_markdown(
    output_dir: Path,
    model_name: str,
    primary_metric: str | None,
    primary_metric_value: float | None,
    metrics: dict[str, float],
) -> Path:
    markdown_path = output_dir / "eval.md"
    lines = [
        "# Evaluation Result",
        "",
        f"- model: `{model_name}`",
        f"- primary_metric: `{primary_metric}`",
        f"- primary_metric_value: `{primary_metric_value}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in sorted(metrics):
        lines.append(f"| `{key}` | {metrics[key]} |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def _collect_runtime_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "package_versions": get_versions(),
        "cuda": {
            "is_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device_count": torch.cuda.device_count(),
            "devices": [],
            "current_device_index": None,
            "current_device_name": None,
        },
    }

    if torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        environment["cuda"]["current_device_index"] = current_index
        environment["cuda"]["current_device_name"] = torch.cuda.get_device_name(current_index)
        environment["cuda"]["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)} for index in range(torch.cuda.device_count())
        ]

    return environment


def main() -> None:
    args = parse_args()
    model = load_model(args)
    output_dir = _resolve_output_dir(args)
    prompt_info = _resolve_effective_prompts(model, args)
    _print_effective_prompts(prompt_info)
    eval_plan = _collect_eval_plan(args)
    _print_eval_plan(eval_plan)

    evaluators = build_evaluators(args)
    sequential = SequentialEvaluator(
        evaluators,
        main_score_function=lambda scores: float(np.mean(scores)),
    )

    eval_started_at_utc = datetime.now(timezone.utc)
    eval_start = time.perf_counter()
    metrics = _normalize_metrics(sequential(model, output_path=str(output_dir)))
    eval_duration_seconds = time.perf_counter() - eval_start
    eval_finished_at_utc = datetime.now(timezone.utc)

    ndcg_metrics = {key: value for key, value in metrics.items() if "ndcg@10" in key}
    if not ndcg_metrics:
        logger.warning("No ndcg@10 metrics were found in the evaluation output.")

    primary_metric = sequential.primary_metric
    primary_metric_value = float(metrics[primary_metric]) if primary_metric in metrics else None
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name_or_path": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "attn_implementation": _resolve_attn_implementation(args),
            "trust_remote_code": args.trust_remote_code,
            "max_seq_length": model.max_seq_length,
            "similarity_fn_name": model.similarity_fn_name,
        },
        "environment": _collect_runtime_environment(),
        "cli_args": vars(args),
        "prompts": prompt_info,
        "evaluation_plan": eval_plan,
        "evaluation": {
            "batch_size": args.batch_size,
            "duration_seconds_excluding_dataset_load": eval_duration_seconds,
            "started_at_utc": eval_started_at_utc.isoformat(),
            "finished_at_utc": eval_finished_at_utc.isoformat(),
            "evaluated_at_utc": eval_finished_at_utc.isoformat(),
        },
        "primary_metric": primary_metric,
        "primary_metric_value": primary_metric_value,
        "metrics": metrics,
        "ndcg_at_10_metrics": ndcg_metrics,
    }
    json_path = _write_eval_json(output_dir, payload)
    csv_path = _write_eval_csv(output_dir, metrics)
    markdown_path = _write_eval_markdown(output_dir, args.model, primary_metric, primary_metric_value, metrics)
    output_paths = {
        "eval_json": str(json_path),
        "eval_csv": str(csv_path),
        "eval_md": str(markdown_path),
    }

    print(
        json.dumps(
            {"primary_metric": primary_metric, "metrics": ndcg_metrics, "output_files": output_paths},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
