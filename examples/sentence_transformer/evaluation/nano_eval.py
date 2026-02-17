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
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator, NanoEvaluator, SentenceEvaluator, SequentialEvaluator

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
        default="msmarco,nq",
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


def resolve_split_targets(args: argparse.Namespace) -> list[str] | None:
    split_targets = parse_repeated_csv(args.split_target)
    if split_targets:
        return split_targets

    fallback_targets = parse_csv(args.nanobeir_datasets)
    return fallback_targets if fallback_targets else None


def resolve_nanobeir_dataset_ids(args: argparse.Namespace) -> list[str]:
    collection_names = resolve_collections(args.collection)

    dataset_ids: list[str] = []
    for collection_name in collection_names:
        dataset_ids.extend(COLLECTION_DATASET_IDS[collection_name])

    if args.nanobeir_dataset_id is not None:
        dataset_ids.extend(parse_csv(args.nanobeir_dataset_id))

    if not dataset_ids:
        dataset_ids = ["sentence-transformers/NanoBEIR-en"]

    deduped: list[str] = []
    seen: set[str] = set()
    for dataset_id in dataset_ids:
        if dataset_id in seen:
            continue
        seen.add(dataset_id)
        deduped.append(dataset_id)
    return deduped


def dataset_id_to_prefix(dataset_id: str) -> str:
    if dataset_id in NANOBEIR_DATASET_ID_TO_HUMAN_READABLE_PREFIX:
        return NANOBEIR_DATASET_ID_TO_HUMAN_READABLE_PREFIX[dataset_id]
    return dataset_id.replace("/", "__").replace("-", "_")


def build_evaluators(args: argparse.Namespace) -> list[Any]:
    split_targets = resolve_split_targets(args)
    nanobeir_dataset_ids = resolve_nanobeir_dataset_ids(args)
    extra_splits = parse_csv(args.extra_splits)

    evaluators: list[Any] = []
    for dataset_id in nanobeir_dataset_ids:
        nanobeir_evaluator = NanoBEIREvaluator(
            dataset_names=split_targets,
            dataset_id=dataset_id,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress,
            write_csv=False,
            query_prompts=args.query_prompt,
            corpus_prompts=args.corpus_prompt,
        )
        evaluators.append(PrefixedEvaluator(nanobeir_evaluator, prefix=dataset_id_to_prefix(dataset_id)))

    if args.extra_dataset_id is not None:
        extra_evaluator = NanoEvaluator(
            dataset_names=extra_splits or None,
            dataset_id=args.extra_dataset_id,
            batch_size=args.batch_size,
            show_progress_bar=args.show_progress,
            write_csv=False,
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


def load_model(args: argparse.Namespace) -> SentenceTransformer:
    attn_implementation = args.attn_implementation
    if args.flash_attn2:
        if attn_implementation and attn_implementation != "flash_attention_2":
            raise ValueError("Both --flash-attn2 and --attn-implementation were provided with conflicting values.")
        attn_implementation = "flash_attention_2"

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


def _print_effective_prompts(model: SentenceTransformer, args: argparse.Namespace) -> None:
    query_prompt, query_source = _resolve_effective_query_prompt(model, args.query_prompt)
    corpus_prompt, corpus_source = _resolve_effective_corpus_prompt(model, args.corpus_prompt)

    if query_prompt is not None:
        print(f"query prompt [{query_source}]: {query_prompt!r}")
    else:
        print("query prompt [none]: None")

    if corpus_prompt is not None:
        print(f"doc prompt [{corpus_source}]: {corpus_prompt!r}")
    else:
        print("doc prompt [none]: None")


def _print_eval_plan(args: argparse.Namespace) -> None:
    split_targets = resolve_split_targets(args)
    dataset_ids = resolve_nanobeir_dataset_ids(args)
    print(f"NanoBEIR dataset ids: {dataset_ids}")
    print(f"split targets: {split_targets if split_targets is not None else 'all'}")
    if args.extra_dataset_id is not None:
        extra_splits = parse_csv(args.extra_splits)
        print(f"extra dataset id: {args.extra_dataset_id}")
        print(f"extra splits: {extra_splits if extra_splits else 'all'}")


def main() -> None:
    args = parse_args()
    model = load_model(args)
    _print_effective_prompts(model, args)
    _print_eval_plan(args)

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
