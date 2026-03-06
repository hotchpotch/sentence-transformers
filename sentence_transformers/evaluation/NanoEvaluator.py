from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation._nano_utils import _GenericNanoDatasetMixin
from sentence_transformers.evaluation.NanoBEIREvaluator import NanoBEIREvaluator
from sentence_transformers.similarity_functions import SimilarityFunction


class NanoEvaluator(_GenericNanoDatasetMixin, NanoBEIREvaluator):
    """
    Generic evaluator for Nano-style Information Retrieval datasets on Hugging Face.

    This evaluator supports direct split names as well as short dataset names that are
    expanded through ``dataset_name_to_human_readable`` and ``split_prefix``.
    """

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        write_predictions: bool = False,
        dataset_name_to_human_readable: Mapping[str, str] | None = None,
        split_prefix: str = "",
        strict_dataset_name_validation: bool = False,
        auto_expand_splits_when_dataset_names_none: bool = True,
        corpus_subset_name: str = "corpus",
        queries_subset_name: str = "queries",
        qrels_subset_name: str = "qrels",
        name: str | None = None,
    ) -> None:
        self._initialize_generic_nano_state(
            dataset_id=dataset_id,
            dataset_name_to_human_readable=dataset_name_to_human_readable,
            split_prefix=split_prefix,
            strict_dataset_name_validation=strict_dataset_name_validation,
            auto_expand_splits_when_dataset_names_none=auto_expand_splits_when_dataset_names_none,
            corpus_subset_name=corpus_subset_name,
            queries_subset_name=queries_subset_name,
            qrels_subset_name=qrels_subset_name,
            name=name,
        )
        super().__init__(
            dataset_names=self._resolve_dataset_names(dataset_names),
            dataset_id=dataset_id,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
            write_predictions=write_predictions,
        )

    def get_config_dict(self) -> dict[str, Any]:
        return self._get_generic_config_dict()
