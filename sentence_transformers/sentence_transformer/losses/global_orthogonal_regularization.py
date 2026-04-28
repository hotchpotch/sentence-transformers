from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer import SentenceTransformer
from sentence_transformers.util import cos_sim


class GlobalOrthogonalRegularizationLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        similarity_fct=cos_sim,
        mode: Literal["gemma", "original"] = "gemma",
        mean_weight: float | None = None,
        second_moment_weight: float | None = None,
        aggregation: Literal["mean", "sum"] = "sum",
        second_moment_threshold: Literal["auto", "dimension"] | float | None = "auto",
        embedding_indices: tuple[int, ...] | None = None,
    ) -> None:
        """
        Global Orthogonal Regularization (GOR) Loss that encourages embeddings to be well-distributed
        in the embedding space by penalizing high mean similarities and high second moments of similarities
        across unrelated inputs.

        The loss consists of two terms:

        1. Mean term: Penalizes when the mean similarity across unrelated embeddings is high
        2. Second moment term: Penalizes when the second moment of similarities is high

        A high second moment indicates that some embeddings have very high similarities, suggesting clustering
        or concentration in certain regions of the embedding space. A low second moment indicates that
        similarities are more uniformly distributed.

        The loss is called independently on each input column (e.g., queries and passages) and combines the results
        using either mean or sum aggregation. This is why the loss can be used on any dataset configuration
        (e.g., single inputs, pairs, triplets, etc.).

        It's recommended to combine this loss with a primary loss function, such as :class:`MultipleNegativesRankingLoss`.

        Args:
            model: SentenceTransformer model
            similarity_fct: Function to compute similarity between embeddings (default: cosine similarity)
            mode: Which GOR variant to compute. ``"gemma"`` computes the EmbeddingGemma-style spread-out loss
                independently for each input column using off-diagonal similarities. ``"original"`` computes the
                original GOR objective on paired non-matching embeddings from each selected input-column pair.
            mean_weight: Weight for the mean term loss component. If None, defaults to 0.0 for ``"gemma"`` and 1.0
                for ``"original"``. 0 can be used to disable this term.
            second_moment_weight: Weight for the second moment term loss component. If None, defaults to 1.0.
            aggregation: How to combine losses across input columns. Either "mean" or "sum" (default: "sum").
                The EmbeddingGemma paper uses "sum".
            second_moment_threshold: Threshold to subtract from the second moment with a ReLU. If "auto", this is
                None for ``"gemma"`` (raw second moment) and "dimension" for ``"original"`` (1/d threshold).
                If None, the raw second moment is used. If "dimension", uses the original GOR threshold 1/d.
            embedding_indices: Optionally select the embedding columns used for GOR. For ``mode="original"``,
                this must select at least two columns; GOR is computed over all unordered selected-column pairs.
                For ``mode="gemma"``, this can
                restrict spread-out regularization to selected columns, e.g. ``(0, 1)`` for query and positive
                columns when the wrapped loss receives ``(anchor, positive, negative)`` inputs.

        References:
            - For further details, see: https://huggingface.co/papers/1708.06320 or https://huggingface.co/papers/2509.20354.
              The latter paper uses the equivalent of GOR with ``mean_weight=0.0`` and ``aggregation="sum"``.

        Inputs:
            +-------+--------+
            | Texts | Labels |
            +=======+========+
            | any   | none   |
            +-------+--------+

        Example:
            ::

                from datasets import Dataset
                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from sentence_transformers.sentence_transformer.losses import (
                    GlobalOrthogonalRegularizationWrapperLoss,
                    MultipleNegativesRankingLoss,
                )

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })

                base_loss = MultipleNegativesRankingLoss(model)
                loss = GlobalOrthogonalRegularizationWrapperLoss(model, base_loss)
                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            Alternatively, you can use multi-task learning to train with both losses:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
                from sentence_transformers.sentence_transformer.losses import GlobalOrthogonalRegularizationLoss, MultipleNegativesRankingLoss
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                mnrl_loss = MultipleNegativesRankingLoss(model)
                gor_loss = GlobalOrthogonalRegularizationLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset={"main": train_dataset, "gor": train_dataset},
                    loss={"main": mnrl_loss, "gor": gor_loss},
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        if mode not in ["gemma", "original"]:
            raise ValueError(f"mode must be 'gemma' or 'original', got '{mode}'")
        self.mode = mode
        if mean_weight is None:
            mean_weight = 0.0 if mode == "gemma" else 1.0
        if second_moment_weight is None:
            second_moment_weight = 1.0
        if second_moment_threshold == "auto":
            second_moment_threshold = None if mode == "gemma" else "dimension"
        self.mean_weight = mean_weight
        self.second_moment_weight = second_moment_weight
        if not mean_weight and not second_moment_weight:
            raise ValueError("At least one of mean_weight or second_moment_weight must be non-zero")
        if aggregation not in ["mean", "sum"]:
            raise ValueError(f"aggregation must be 'mean' or 'sum', got '{aggregation}'")
        self.aggregation = aggregation
        self.second_moment_threshold = second_moment_threshold
        self.embedding_indices = embedding_indices

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings)

    def compute_loss_from_embeddings(
        self, embeddings: list[Tensor], labels: Tensor | None = None
    ) -> dict[str, Tensor]:
        """
        Compute the GOR loss from pre-computed embeddings.

        Args:
            embeddings: List of embedding tensors, one for each input column (e.g., [queries, passages])
            labels: Not used, kept for compatibility

        Returns:
            Dictionary containing the weighted mean term and second moment term losses
        """
        if self.mode == "original":
            if self.embedding_indices is None and len(embeddings) == 2:
                selected_embeddings = embeddings
            else:
                selected_embeddings = self._select_embeddings(embeddings)
            if len(selected_embeddings) < 2:
                raise ValueError(
                    'GlobalOrthogonalRegularizationLoss with mode="original" requires at least two embedding '
                    "columns. If the wrapped loss receives additional columns, pass embedding_indices to choose "
                    "the columns to regularize."
                )
            pair_terms = [
                self.compute_original_gor(selected_embeddings[left], selected_embeddings[right])
                for left in range(len(selected_embeddings))
                for right in range(left + 1, len(selected_embeddings))
            ]
            mean_terms, second_moment_terms = zip(*pair_terms)
        else:
            selected_embeddings = self._select_embeddings(embeddings)
            mean_terms, second_moment_terms = zip(
                *[self.compute_gemma_gor(embedding) for embedding in selected_embeddings]
            )
        results = {}
        if self.mean_weight:
            stacked_mean = torch.stack(mean_terms)
            aggregated_mean = stacked_mean.sum() if self.aggregation == "sum" else stacked_mean.mean()
            results["gor_mean"] = self.mean_weight * aggregated_mean
        if self.second_moment_weight:
            stacked_second_moment = torch.stack(second_moment_terms)
            aggregated_second_moment = (
                stacked_second_moment.sum() if self.aggregation == "sum" else stacked_second_moment.mean()
            )
            results["gor_second_moment"] = self.second_moment_weight * aggregated_second_moment
        return results

    def _select_embeddings(self, embeddings: list[Tensor]) -> list[Tensor]:
        if self.embedding_indices is None:
            return embeddings
        try:
            return [embeddings[index] for index in self.embedding_indices]
        except IndexError as exc:
            raise ValueError(
                f"GlobalOrthogonalRegularizationLoss received embedding_indices={self.embedding_indices}, "
                f"but only {len(embeddings)} embedding columns are available."
            ) from exc

    def compute_gor(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        return self.compute_gemma_gor(embeddings)

    def compute_gemma_gor(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the EmbeddingGemma-style spread-out terms for a batch of embeddings.

        The GOR loss encourages embeddings to be well-distributed by:
        1. Mean term (M_1^2): Penalizes high mean similarity, pushing embeddings apart
        2. Second moment term: Penalizes high off-diagonal similarity second moment

        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim)

        Returns:
            Tuple of (mean_term, second_moment_term) losses (unweighted)
        """
        batch_size = embeddings.size(0)
        if batch_size < 2:
            raise ValueError("GlobalOrthogonalRegularizationLoss requires at least 2 embeddings per input column.")
        hidden_dim = embeddings.size(1)
        embeddings = embeddings.float()

        # Compute GOR statistics in fp32 even under mixed-precision training.
        with self._disable_autocast(embeddings):
            sim_matrix = self.similarity_fct(embeddings, embeddings)
        sim_matrix = sim_matrix.masked_fill(torch.eye(batch_size, device=sim_matrix.device, dtype=torch.bool), 0.0)
        num_off_diagonal = batch_size * (batch_size - 1)

        # Mean term: M_1^2 where M_1 = mean of off-diagonal similarities
        # Penalizes high similarities across inputs from the same column (e.g., queries vs other queries)
        mean_term = (sim_matrix.sum() / num_off_diagonal).pow(2)

        # Second moment term: raw M_2 by default, matching EmbeddingGemma. The original GOR thresholded variant
        # can be selected with second_moment_threshold="dimension" or a float threshold.
        second_moment = sim_matrix.pow(2).sum() / num_off_diagonal
        if self.second_moment_threshold is None:
            second_moment_term = second_moment
        else:
            threshold = 1.0 / hidden_dim if self.second_moment_threshold == "dimension" else self.second_moment_threshold
            second_moment_term = torch.relu(second_moment - threshold)

        return mean_term, second_moment_term

    def compute_original_gor(self, anchors: Tensor, negatives: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the original GOR terms on paired non-matching embeddings.

        This matches the original paper and reference HardNet implementation: for each row ``i``, compute the
        similarity between ``anchors[i]`` and ``negatives[i]``, then apply ``M_1^2 + max(0, M_2 - 1/d)`` by default.
        """
        if anchors.shape != negatives.shape:
            raise ValueError(
                "GlobalOrthogonalRegularizationLoss original mode requires anchor and negative embeddings with "
                f"the same shape, got {tuple(anchors.shape)} and {tuple(negatives.shape)}."
            )
        if anchors.dim() != 2:
            raise ValueError("GlobalOrthogonalRegularizationLoss requires 2D embedding tensors.")
        batch_size = anchors.size(0)
        if batch_size < 1:
            raise ValueError("GlobalOrthogonalRegularizationLoss original mode requires at least 1 embedding pair.")
        hidden_dim = anchors.size(1)
        anchors = anchors.float()
        negatives = negatives.float()

        with self._disable_autocast(anchors):
            sim_matrix = self.similarity_fct(anchors, negatives)
        if sim_matrix.shape != (batch_size, batch_size):
            raise ValueError(
                "GlobalOrthogonalRegularizationLoss original mode expects the similarity function to return a "
                f"{batch_size}x{batch_size} pairwise matrix, got {tuple(sim_matrix.shape)}."
            )
        paired_similarities = sim_matrix.diagonal()
        mean_term = paired_similarities.mean().pow(2)
        second_moment = paired_similarities.pow(2).mean()
        if self.second_moment_threshold is None:
            second_moment_term = second_moment
        else:
            threshold = 1.0 / hidden_dim if self.second_moment_threshold == "dimension" else self.second_moment_threshold
            second_moment_term = torch.relu(second_moment - threshold)

        return mean_term, second_moment_term

    @staticmethod
    def _disable_autocast(tensor: Tensor) -> AbstractContextManager:
        if tensor.device.type in {"cuda", "cpu", "xpu", "hpu"}:
            return torch.amp.autocast(device_type=tensor.device.type, enabled=False)
        return nullcontext()

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "similarity_fct": self.similarity_fct.__name__,
            "mode": self.mode,
            "mean_weight": self.mean_weight,
            "second_moment_weight": self.second_moment_weight,
            "aggregation": self.aggregation,
            "second_moment_threshold": self.second_moment_threshold,
            "embedding_indices": self.embedding_indices,
        }

    @property
    def citation(self) -> str:
        return """
@misc{zhang2017learningspreadoutlocalfeature,
      title={Learning Spread-out Local Feature Descriptors},
      author={Xu Zhang and Felix X. Yu and Sanjiv Kumar and Shih-Fu Chang},
      year={2017},
      eprint={1708.06320},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/1708.06320},
}
"""
