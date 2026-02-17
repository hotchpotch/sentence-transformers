from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation.NanoEvaluator import NanoEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]

DATASET_NAME_TO_HUMAN_READABLE: dict[str, str] = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class NanoBEIREvaluator(NanoEvaluator):
    """Evaluate a SentenceTransformer model on NanoBEIR datasets.

    This class preserves the historical NanoBEIR short-name API
    (for example ``msmarco``, ``nq``) while delegating the dataset loading
    and aggregation mechanics to :class:`~sentence_transformers.evaluation.NanoEvaluator`.
    """

    def __init__(
        self,
        dataset_names: list[DatasetNameType | str] | None = None,
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
    ) -> None:
        if dataset_names is None:
            dataset_names = list(DATASET_NAME_TO_HUMAN_READABLE.keys())

        super().__init__(
            dataset_names=[str(name) for name in dataset_names],
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
            dataset_name_to_human_readable=DATASET_NAME_TO_HUMAN_READABLE,
            split_prefix="Nano",
            strict_dataset_name_validation=True,
            auto_expand_splits_when_dataset_names_none=False,
            name="NanoBEIR",
        )

    def _validate_dataset_names(self) -> None:
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")

        missing_datasets = [
            dataset_name
            for dataset_name in self.dataset_names
            if dataset_name.lower() not in DATASET_NAME_TO_HUMAN_READABLE
        ]
        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} are not valid NanoBEIR datasets. "
                f"Valid dataset names are: {list(DATASET_NAME_TO_HUMAN_READABLE.keys())}"
            )

    def get_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
        }
        config_dict_candidate_keys = ["truncate_dim", "query_prompts", "corpus_prompts"]
        for key in config_dict_candidate_keys:
            value = getattr(self, key)
            if value is not None:
                config_dict[key] = value
        return config_dict
