from __future__ import annotations

"""Evaluate Nano benchmark datasets with embedding models, cross-encoders, or sparse encoders.

This script supports three evaluation modes:
1. ``--model-type embedding``: uses ``SentenceTransformer`` + Nano IR evaluators.
2. ``--model-type cross-encoder``: uses ``CrossEncoder`` + Nano reranking evaluators.
3. ``--model-type sparse-encoder``: uses ``SparseEncoder`` + Sparse Nano evaluators.

Dataset selection:
- ``-c/--collection NanoBEIR`` -> ``sentence-transformers/NanoBEIR-en``.
- ``-c/--collection MNanoBEIR`` -> multilingual NanoBEIR collection.
- ``-d/--dataset-id ...`` -> generic Nano datasets, e.g. ``hotchpotch/NanoMIRACL``.
- ``-s/--split-target ...`` -> split filter for NanoBEIR collections only.
  If omitted, NanoBEIR runs all available benchmark splits.

Cache behavior:
- Cache key is file existence only (split JSON path).
- If you change prompts, dtype, batch size, attention, or evaluator settings,
  remove cached files or use ``--override`` to force re-evaluation.

Offline execution tip:
- If datasets/models are already cached locally, setting
  ``HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`` reduces API traffic and stabilizes
  large runs.

Quick examples:
1) Embedding model on all NanoBEIR-en splits
   ``uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
   --model-type embedding --model intfloat/multilingual-e5-small -c NanoBEIR``

2) Embedding model on multilingual NanoBEIR + extra datasets
   ``uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
   --model intfloat/multilingual-e5-large -c MNanoBEIR \\
   -d hotchpotch/NanoMIRACL -d hotchpotch/NanoCodeSearchNet \\
   --query-prompt "query: " --corpus-prompt "passage: "``

3) Cross-encoder reranking on NanoBEIR-en with bm25 candidates
   ``uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
   --model-type cross-encoder --model cross-encoder/ms-marco-MiniLM-L6-v2 \\
   -c NanoBEIR --candidate-subset-name bm25``

4) Cross-encoder reranking with custom candidate subset name
   ``uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
   --model-type cross-encoder --model your-org/your-reranker \\
   -d hotchpotch/NanoCodeSearchNet --candidate-subset-name dense``

5) Sparse encoder evaluation on NanoBEIR
   ``uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
   --model-type sparse-encoder \\
   --model sparse-encoder/example-inference-free-splade-distilbert-base-uncased-nq \\
   -c NanoBEIR --query-prompt "query: " --corpus-prompt "passage: "``
"""

import argparse
import importlib
import json
import logging
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderNanoEvaluator
from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator
from sentence_transformers.evaluation import NanoBEIREvaluator, NanoEvaluator
from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.evaluation.NanoBEIREvaluator import DATASET_NAME_TO_HUMAN_READABLE
from sentence_transformers.model_card import get_versions
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator, SparseNanoEvaluator
from sentence_transformers.util import is_datasets_available

logger = logging.getLogger(__name__)

TIMING_KEYS = [
    "query_embedding_seconds",
    "corpus_embedding_seconds",
    "score_and_topk_seconds",
    "metric_compute_seconds",
    "pure_compute_seconds",
]

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

COLLECTION_TO_EVALUATOR_KIND: dict[str, str] = {
    "nanobeir": "nanobeir",
    "mnanobeir": "nanobeir",
}

HELP_EPILOG = """Examples:
  # 1) Embedding evaluation on all NanoBEIR-en splits
  uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
    --model intfloat/multilingual-e5-small -c NanoBEIR

  # 2) Embedding evaluation on MNanoBEIR + custom datasets
  uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
    --model intfloat/multilingual-e5-large -c MNanoBEIR \\
    -d hotchpotch/NanoMIRACL -d hotchpotch/NanoCodeSearchNet \\
    --query-prompt "query: " --corpus-prompt "passage: "

  # 3) Cross-encoder reranking on NanoBEIR-en
  uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
    --model-type cross-encoder --model cross-encoder/ms-marco-MiniLM-L6-v2 \\
    -c NanoBEIR --candidate-subset-name bm25

  # 4) Sparse-encoder evaluation on NanoBEIR-en
  uv run --with datasets python examples/sentence_transformer/evaluation/nano_eval.py \\
    --model-type sparse-encoder \\
    --model sparse-encoder/example-inference-free-splade-distilbert-base-uncased-nq \\
    -c NanoBEIR --query-prompt "query: " --corpus-prompt "passage: "

Notes:
  - For more context and usage patterns, read this script's top docstring.
  - If you change parameters, remove cached split JSON files or use --override.
"""


@dataclass(frozen=True)
class EvalTask:
    evaluator_kind: str
    dataset_id: str
    target_name: str
    split_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Nano datasets with embedding, cross-encoder, or sparse-encoder models, "
            "with split-level JSON caching."
        ),
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        default="embedding",
        choices=["embedding", "cross-encoder", "sparse-encoder"],
        help=(
            "Model family for evaluation. "
            "'embedding' uses SentenceTransformer retrieval, "
            "'cross-encoder' uses reranking evaluators, "
            "'sparse-encoder' uses SparseEncoder retrieval evaluators."
        ),
    )
    parser.add_argument("--model", required=True, help="Model name on Hugging Face Hub or local path.")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model dtype for loading/inference.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional Transformers attention implementation (e.g., 'flash_attention_2').",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        help="Shortcut for '--attn-implementation flash_attention_2'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g., 'cuda', 'cpu'). If omitted, backend auto-selects.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code during model loading.",
    )
    parser.add_argument(
        "--model-max-seq-length",
        type=int,
        default=None,
        help="Optional sequence-length override. If omitted, keep model default.",
    )

    parser.add_argument(
        "-c",
        "--collection",
        action="append",
        default=None,
        help=(
            "Named NanoBEIR collection(s), repeatable or comma-separated. "
            "Supported: NanoBEIR, MNanoBEIR, MultilingualNanoBEIR."
        ),
    )
    parser.add_argument(
        "--nanobeir-dataset-id",
        default=None,
        help=(
            "Optional extra NanoBEIR dataset id(s), repeatable via comma. "
            "Used in addition to --collection. Example: sentence-transformers/NanoBEIR-en."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset-id",
        action="append",
        default=None,
        help=(
            "Optional generic NanoEvaluator dataset id(s), repeatable or comma-separated. "
            "Example: -d hotchpotch/NanoCodeSearchNet,hotchpotch/NanoMIRACL."
        ),
    )
    parser.add_argument(
        "-s",
        "--split-target",
        action="append",
        default=None,
        help=(
            "NanoBEIR split target(s) like msmarco,nq. Repeatable and comma-separated. "
            "Applies to NanoBEIR collection entries only. "
            "If omitted, all NanoBEIR splits are evaluated."
        ),
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument(
        "--candidate-subset-name",
        default="bm25",
        help=(
            "Candidate retrieval subset/config name used by cross-encoder Nano evaluators "
            "(e.g., bm25, dense). Cross-encoder mode only."
        ),
    )
    parser.add_argument("--show-progress", action="store_true", help="Show progress bars during evaluation.")
    parser.add_argument(
        "--query-prompt",
        default=None,
        help="Optional query prefix prompt. Used by embedding/sparse evaluators.",
    )
    parser.add_argument(
        "--corpus-prompt",
        default=None,
        help="Optional corpus/document prefix prompt. Used by embedding/sparse evaluators.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/nano_eval",
        help=("Base output directory. Results are saved to {output-dir}/{model-type}-{model_name}/."),
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-run evaluations even if split JSON cache exists.",
    )
    parser.add_argument(
        "--aggregate-metric",
        default="ndcg@10",
        help="Metric suffix used for per-split and overall aggregation (e.g., ndcg@10, map).",
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


def resolve_nanobeir_split_targets(args: argparse.Namespace) -> list[str] | None:
    split_targets = parse_repeated_csv(args.split_target)
    if split_targets:
        return split_targets

    # Default behavior: evaluate all NanoBEIR splits when split targets are omitted.
    return None


def _resolve_query_splits(dataset_id: str) -> list[str]:
    if not is_datasets_available():
        raise ValueError("datasets is not available. Install it to evaluate Nano datasets.")

    datasets_module = importlib.import_module("datasets")
    get_dataset_split_names = getattr(datasets_module, "get_dataset_split_names")

    split_names = get_dataset_split_names(dataset_id, "queries")
    if not split_names:
        raise ValueError(f"No query splits were found for dataset '{dataset_id}'.")
    return list(split_names)


def resolve_tasks(args: argparse.Namespace) -> list[EvalTask]:
    entries = resolve_collection_dataset_entries(args)
    nanobeir_split_targets = resolve_nanobeir_split_targets(args)
    all_nanobeir_splits = list(DATASET_NAME_TO_HUMAN_READABLE.keys())

    tasks: list[EvalTask] = []
    for evaluator_kind, dataset_id in entries:
        target_name = dataset_id.split("/")[-1]
        if evaluator_kind == "nanobeir":
            split_names = nanobeir_split_targets if nanobeir_split_targets is not None else all_nanobeir_splits
        else:
            split_names = _resolve_query_splits(dataset_id)

        for split_name in split_names:
            tasks.append(
                EvalTask(
                    evaluator_kind=evaluator_kind,
                    dataset_id=dataset_id,
                    target_name=target_name,
                    split_name=split_name,
                )
            )

    deduped: list[EvalTask] = []
    seen: set[tuple[str, str, str]] = set()
    for task in tasks:
        key = (task.evaluator_kind, task.dataset_id, task.split_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(task)

    if not deduped:
        raise ValueError("No evaluation tasks were resolved from the given arguments.")

    return deduped


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


def load_model(args: argparse.Namespace) -> SentenceTransformer | CrossEncoder | SparseEncoder:
    attn_implementation = _resolve_attn_implementation(args)

    if args.model_type == "embedding":
        model_kwargs: dict[str, Any] = {"dtype": _resolve_dtype(args.dtype)}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        model = SentenceTransformer(
            args.model,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            model_kwargs=model_kwargs,
        )
        if args.model_max_seq_length is not None:
            model.max_seq_length = args.model_max_seq_length
        return model

    model_kwargs = {"dtype": _resolve_dtype(args.dtype)}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if args.model_type == "sparse-encoder":
        model = SparseEncoder(
            args.model,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            model_kwargs=model_kwargs,
        )
        if args.model_max_seq_length is not None:
            model.max_seq_length = args.model_max_seq_length
        return model

    return CrossEncoder(
        args.model,
        device=args.device,
        max_length=args.model_max_seq_length,
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


def _resolve_effective_prompts(
    model: SentenceTransformer | CrossEncoder | SparseEncoder, args: argparse.Namespace
) -> dict[str, str | None]:
    if args.model_type == "cross-encoder":
        query_source = "cli(--query-prompt)" if args.query_prompt is not None else "none"
        corpus_source = "cli(--corpus-prompt)" if args.corpus_prompt is not None else "none"
        return {
            "query_prompt": args.query_prompt,
            "query_prompt_source": query_source,
            "corpus_prompt": args.corpus_prompt,
            "corpus_prompt_source": corpus_source,
        }

    if not isinstance(model, SentenceTransformer):
        raise TypeError("Expected SentenceTransformer-compatible model for embedding/sparse prompt resolution.")

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


def _sanitize_output_part(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _resolve_model_output_name(model_name_or_path: str, model_type: str) -> str:
    normalized = model_name_or_path.strip().rstrip("/\\")
    if not normalized:
        return f"{_sanitize_output_part(model_type)}-model"
    model_name = normalized.replace("\\", "/").split("/")[-1]
    safe_name = _sanitize_output_part(model_name)
    name_part = safe_name or "model"
    return f"{_sanitize_output_part(model_type)}-{name_part}"


def _migrate_legacy_dense_output_dir(base_output_dir: Path, model_name_or_path: str, model_type: str) -> None:
    if model_type != "embedding":
        return

    legacy_output_dir = base_output_dir / _resolve_model_output_name(model_name_or_path, "dense")
    embedding_output_dir = base_output_dir / _resolve_model_output_name(model_name_or_path, "embedding")
    if legacy_output_dir.exists() and not embedding_output_dir.exists():
        shutil.move(str(legacy_output_dir), str(embedding_output_dir))
        print(f"migrated legacy output dir: {legacy_output_dir} -> {embedding_output_dir}")


def _resolve_model_output_dir(args: argparse.Namespace) -> Path:
    base_output_dir = Path(args.output_dir)
    _migrate_legacy_dense_output_dir(base_output_dir, args.model, args.model_type)
    output_dir = base_output_dir / _resolve_model_output_name(args.model, args.model_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_split_output_path(model_output_dir: Path, task: EvalTask) -> Path:
    target_dir = model_output_dir / "evals" / _sanitize_output_part(task.target_name)
    target_dir.mkdir(parents=True, exist_ok=True)
    split_file_name = f"{_sanitize_output_part(task.split_name)}.json"
    return target_dir / split_file_name


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _extract_metric_values(metrics: dict[str, float], metric_suffix: str) -> list[float]:
    return [value for key, value in metrics.items() if key.endswith(metric_suffix)]


def _compute_aggregate_metric_value(metrics: dict[str, float], metric_suffix: str) -> float:
    candidates = _extract_metric_values(metrics, metric_suffix)
    if not candidates:
        raise ValueError(
            f"Could not find any metric ending with '{metric_suffix}'. Available metric keys: {sorted(metrics.keys())}"
        )
    return float(np.mean(candidates))


def _collect_runtime_environment() -> dict[str, Any]:
    cuda_info: dict[str, Any] = {
        "is_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": [],
        "current_device_index": None,
        "current_device_name": None,
    }

    if torch.cuda.is_available():
        current_index = torch.cuda.current_device()
        cuda_info["current_device_index"] = current_index
        cuda_info["current_device_name"] = torch.cuda.get_device_name(current_index)
        cuda_info["devices"] = [
            {"index": index, "name": torch.cuda.get_device_name(index)} for index in range(torch.cuda.device_count())
        ]

    return {
        "package_versions": get_versions(),
        "cuda": cuda_info,
    }


def _create_evaluator(
    task: EvalTask, args: argparse.Namespace
) -> InformationRetrievalEvaluator | CrossEncoderRerankingEvaluator:
    if args.model_type == "cross-encoder":
        common_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "show_progress_bar": args.show_progress,
            "write_csv": False,
            "candidate_subset_name": args.candidate_subset_name,
        }
        if task.evaluator_kind == "nanobeir":
            wrapper = CrossEncoderNanoBEIREvaluator(
                dataset_names=[task.split_name],
                dataset_id=task.dataset_id,
                **common_kwargs,
            )
        else:
            wrapper = CrossEncoderNanoEvaluator(
                dataset_names=[task.split_name],
                dataset_id=task.dataset_id,
                **common_kwargs,
            )
        if len(wrapper.evaluators) != 1:
            raise ValueError(f"Expected exactly one CE evaluator for task {task}, got {len(wrapper.evaluators)}")
        return wrapper.evaluators[0]

    if args.model_type == "sparse-encoder":
        if task.evaluator_kind == "nanobeir":
            wrapper = SparseNanoBEIREvaluator(
                dataset_names=[task.split_name],
                dataset_id=task.dataset_id,
                batch_size=args.batch_size,
                show_progress_bar=args.show_progress,
                write_csv=False,
                query_prompts=args.query_prompt,
                corpus_prompts=args.corpus_prompt,
            )
        else:
            wrapper = SparseNanoEvaluator(
                dataset_names=[task.split_name],
                dataset_id=task.dataset_id,
                batch_size=args.batch_size,
                show_progress_bar=args.show_progress,
                write_csv=False,
                query_prompts=args.query_prompt,
                corpus_prompts=args.corpus_prompt,
            )
        if len(wrapper.evaluators) != 1:
            raise ValueError(f"Expected exactly one sparse evaluator for task {task}, got {len(wrapper.evaluators)}")
        return wrapper.evaluators[0]

    common_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "show_progress_bar": args.show_progress,
        "write_csv": False,
        "query_prompts": args.query_prompt,
        "corpus_prompts": args.corpus_prompt,
    }

    if task.evaluator_kind == "nanobeir":
        wrapper = NanoBEIREvaluator(
            dataset_names=[task.split_name],
            dataset_id=task.dataset_id,
            **common_kwargs,
        )
    else:
        wrapper = NanoEvaluator(
            dataset_names=[task.split_name],
            dataset_id=task.dataset_id,
            **common_kwargs,
        )

    if len(wrapper.evaluators) != 1:
        raise ValueError(f"Expected exactly one embedding evaluator for task {task}, got {len(wrapper.evaluators)}")
    return wrapper.evaluators[0]


def _build_model_metadata(
    model: SentenceTransformer | CrossEncoder | SparseEncoder, args: argparse.Namespace
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_type": args.model_type,
        "name_or_path": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": _resolve_attn_implementation(args),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_type in {"embedding", "sparse-encoder"}:
        if not isinstance(model, SentenceTransformer):
            raise TypeError(f"Expected SentenceTransformer-compatible model for model_type='{args.model_type}'")
        metadata["max_seq_length"] = model.max_seq_length
        metadata["similarity_fn_name"] = model.similarity_fn_name
        metadata["max_active_dims"] = (
            getattr(model, "max_active_dims", None) if args.model_type == "sparse-encoder" else None
        )
    else:
        if not isinstance(model, CrossEncoder):
            raise TypeError(f"Expected CrossEncoder model for model_type='{args.model_type}'")
        metadata["max_seq_length"] = model.max_length
        metadata["similarity_fn_name"] = None
        metadata["max_active_dims"] = None
    return metadata


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_or_load_split(
    task: EvalTask,
    model: SentenceTransformer | CrossEncoder | SparseEncoder,
    args: argparse.Namespace,
    prompt_info: dict[str, str | None],
    model_output_dir: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    output_path = _resolve_split_output_path(model_output_dir, task)

    if output_path.exists() and not args.override:
        print(f"cache hit: {task.target_name}/{task.split_name} -> {output_path}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return {
            "cache_hit": True,
            "output_path": str(output_path),
            "payload": payload,
        }

    print(f"evaluating: {task.target_name}/{task.split_name}")
    evaluator = _create_evaluator(task, args)

    eval_started_at_utc = datetime.now(timezone.utc)
    eval_start = time.perf_counter()
    if args.model_type == "cross-encoder":
        if not isinstance(model, CrossEncoder):
            raise TypeError("Cross-encoder mode requires a CrossEncoder model instance.")
        ce_evaluator = cast(CrossEncoderRerankingEvaluator, evaluator)
        metrics = _normalize_metrics(ce_evaluator(model, output_path=None))
    elif args.model_type == "sparse-encoder":
        if not isinstance(model, SparseEncoder):
            raise TypeError("sparse-encoder mode requires a SparseEncoder model instance.")
        ir_evaluator = cast(InformationRetrievalEvaluator, evaluator)
        metrics = _normalize_metrics(ir_evaluator(model, output_path=None))
    else:
        if not isinstance(model, SentenceTransformer):
            raise TypeError("Embedding mode requires a SentenceTransformer model instance.")
        ir_evaluator = cast(InformationRetrievalEvaluator, evaluator)
        metrics = _normalize_metrics(ir_evaluator(model, output_path=None))
    eval_elapsed_seconds = time.perf_counter() - eval_start
    eval_finished_at_utc = datetime.now(timezone.utc)

    timing: dict[str, float] = {key: 0.0 for key in TIMING_KEYS}
    timing["pure_compute_seconds"] = float(eval_elapsed_seconds)
    if hasattr(evaluator, "last_timing"):
        last_timing = getattr(evaluator, "last_timing")
        if isinstance(last_timing, dict):
            for key, value in last_timing.items():
                if key in TIMING_KEYS and isinstance(value, (int, float)):
                    timing[key] = float(value)
            if "pure_compute_seconds" not in last_timing:
                timing["pure_compute_seconds"] = float(eval_elapsed_seconds)

    aggregate_metric_value = _compute_aggregate_metric_value(metrics, args.aggregate_metric)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": _build_model_metadata(model, args),
        "environment": environment,
        "prompts": prompt_info,
        "target": {
            "dataset_id": task.dataset_id,
            "target_name": task.target_name,
            "evaluator_kind": task.evaluator_kind,
            "split_name": task.split_name,
        },
        "config": {
            "batch_size": args.batch_size,
            "aggregate_metric": args.aggregate_metric,
            "show_progress": args.show_progress,
            "candidate_subset_name": args.candidate_subset_name if args.model_type == "cross-encoder" else None,
        },
        "evaluation": {
            "started_at_utc": eval_started_at_utc.isoformat(),
            "finished_at_utc": eval_finished_at_utc.isoformat(),
            # Keep for backward-compatible consumers that read `evaluated_at_utc`.
            "evaluated_at_utc": eval_finished_at_utc.isoformat(),
            "duration_seconds_excluding_dataset_load": float(timing["pure_compute_seconds"]),
            "aggregate_metric": args.aggregate_metric,
            "aggregate_metric_value": aggregate_metric_value,
            "timing": timing,
        },
        "metrics": metrics,
    }
    _write_json(output_path, payload)

    return {
        "cache_hit": False,
        "output_path": str(output_path),
        "payload": payload,
    }


def _build_all_payload(
    model: SentenceTransformer | CrossEncoder | SparseEncoder,
    args: argparse.Namespace,
    prompt_info: dict[str, str | None],
    environment: dict[str, Any],
    split_results: list[dict[str, Any]],
) -> dict[str, Any]:
    split_entries: list[dict[str, Any]] = []
    aggregate_metric_values: list[float] = []

    timing_totals_all: dict[str, float] = {key: 0.0 for key in TIMING_KEYS}
    timing_totals_this_run: dict[str, float] = {key: 0.0 for key in TIMING_KEYS}

    for result in split_results:
        payload = result["payload"]
        cache_hit = bool(result["cache_hit"])
        evaluation = payload.get("evaluation", {})
        timing = evaluation.get("timing", {})
        aggregate_metric_value = evaluation.get("aggregate_metric_value")

        if isinstance(aggregate_metric_value, (int, float)):
            aggregate_metric_values.append(float(aggregate_metric_value))

        for key in TIMING_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                float_value = float(value)
                timing_totals_all[key] += float_value
                if not cache_hit:
                    timing_totals_this_run[key] += float_value

        split_entry = {
            "target_name": payload.get("target", {}).get("target_name"),
            "dataset_id": payload.get("target", {}).get("dataset_id"),
            "split_name": payload.get("target", {}).get("split_name"),
            "evaluator_kind": payload.get("target", {}).get("evaluator_kind"),
            "cache_hit": cache_hit,
            "result_path": result["output_path"],
            "aggregate_metric": evaluation.get("aggregate_metric"),
            "aggregate_metric_value": aggregate_metric_value,
            "duration_seconds_excluding_dataset_load": evaluation.get("duration_seconds_excluding_dataset_load"),
            "evaluated_at_utc": evaluation.get("evaluated_at_utc"),
        }
        split_entries.append(split_entry)

    split_entries.sort(key=lambda item: (str(item["target_name"]), str(item["split_name"])))

    target_buckets: dict[str, list[float]] = {}
    for entry in split_entries:
        value = entry.get("aggregate_metric_value")
        if not isinstance(value, (int, float)):
            continue
        target_name = str(entry["target_name"])
        target_buckets.setdefault(target_name, []).append(float(value))

    target_summaries: dict[str, dict[str, Any]] = {}
    for target_name, values in sorted(target_buckets.items()):
        target_summaries[target_name] = {
            "split_count": len(values),
            "aggregate_metric": args.aggregate_metric,
            "aggregate_metric_mean": float(np.mean(values)),
        }

    all_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": _build_model_metadata(model, args),
        "environment": environment,
        "cli_args": vars(args),
        "prompts": prompt_info,
        "aggregate_metric": args.aggregate_metric,
        "totals": {
            "target_count": len({entry["target_name"] for entry in split_entries}),
            "split_count": len(split_entries),
            "aggregate_metric_mean": float(np.mean(aggregate_metric_values)) if aggregate_metric_values else None,
            "cache_hit_count": sum(1 for entry in split_entries if entry["cache_hit"]),
            "evaluated_count": sum(1 for entry in split_entries if not entry["cache_hit"]),
            "timing_seconds_this_run": timing_totals_this_run,
            "timing_seconds_all_splits": timing_totals_all,
            "total_duration_seconds_excluding_dataset_load": timing_totals_this_run["pure_compute_seconds"],
            "total_duration_seconds_excluding_dataset_load_all_splits": timing_totals_all["pure_compute_seconds"],
        },
        "target_summaries": target_summaries,
        "splits": split_entries,
    }
    return all_payload


def _print_eval_plan(tasks: list[EvalTask]) -> None:
    by_target: dict[str, list[str]] = {}
    for task in tasks:
        by_target.setdefault(task.target_name, []).append(task.split_name)

    print(f"resolved tasks: {len(tasks)} splits across {len(by_target)} targets")
    for target_name, split_names in sorted(by_target.items()):
        print(f"- {target_name}: {len(split_names)} splits")


def main() -> None:
    args = parse_args()
    model = load_model(args)
    prompt_info = _resolve_effective_prompts(model, args)
    _print_effective_prompts(prompt_info)

    tasks = resolve_tasks(args)
    _print_eval_plan(tasks)

    model_output_dir = _resolve_model_output_dir(args)
    environment = _collect_runtime_environment()

    split_results: list[dict[str, Any]] = []
    for task in tasks:
        split_results.append(_run_or_load_split(task, model, args, prompt_info, model_output_dir, environment))

    all_payload = _build_all_payload(model, args, prompt_info, environment, split_results)
    all_output_path = model_output_dir / "all.json"
    _write_json(all_output_path, all_payload)

    print(
        json.dumps(
            {
                "aggregate_metric": args.aggregate_metric,
                "aggregate_metric_mean": all_payload["totals"]["aggregate_metric_mean"],
                "total_duration_seconds_excluding_dataset_load": all_payload["totals"][
                    "total_duration_seconds_excluding_dataset_load"
                ],
                "all_json": str(all_output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
