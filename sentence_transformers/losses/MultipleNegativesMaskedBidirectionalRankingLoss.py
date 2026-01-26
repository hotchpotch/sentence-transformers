from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class MultipleNegativesMaskedBidirectionalRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.01,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        margin: float = 0.1,
        hard_negative_margin: float = 0.1,
        gather_across_devices: bool = False,
    ) -> None:
        """
        Masked InfoNCE loss that uses three pools in the denominator: q->d, q->q, and d->d.

        This variant expands the denominator with query-query and document-document negatives and applies a
        margin-based mask to suppress likely false negatives. The document-document term only uses positives; hard
        negatives are excluded from d->d.

        Args:
            model: SentenceTransformer model
            temperature: Temperature parameter to scale the similarities. The internal scale is derived as
                ``scale = 1 / temperature``, so temperature=0.01 is equivalent to scale=100.0.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            margin: Margin for masking in-batch candidates. Candidates with similarity > s_pos + margin are masked.
            hard_negative_margin: Margin for masking hard negatives. A value <= 0 disables hard negative filtering.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets
            2. Optional negatives are supported as hard negatives (additional documents).

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Notes:
            - Optional negatives are treated as additional documents for the q->d term.
            - The d->d term uses only positives (hard negatives are excluded).
            - Masking is applied in similarity space (pre-temperature scaling).

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesMaskedBidirectionalRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = temperature
        self.similarity_fct = similarity_fct
        self.margin = margin
        self.hard_negative_margin = hard_negative_margin
        self.gather_across_devices = gather_across_devices

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embeddings, got {len(embeddings)}")

        queries = embeddings[0]
        docs_pos = embeddings[1]
        docs_neg = embeddings[2:]
        batch_size = queries.size(0)
        offset = 0

        if self.gather_across_devices:
            queries = all_gather_with_grad(queries)
            docs_pos = all_gather_with_grad(docs_pos)
            docs_neg = [all_gather_with_grad(doc) for doc in docs_neg]
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        total_size = queries.size(0)
        docs_all = torch.cat([docs_pos] + docs_neg, dim=0) if docs_neg else docs_pos

        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)
        local_queries = queries[local_indices]
        local_docs = docs_pos[local_indices]

        sim_qd = self.similarity_fct(local_queries, docs_all)
        row_indices = torch.arange(batch_size, device=queries.device)
        pos_scores = sim_qd[row_indices, local_indices]
        threshold = pos_scores[:, None] + self.margin

        # Mask for off-diagonal entries (exclude self)
        mask_offdiag = torch.ones(batch_size, total_size, dtype=torch.bool, device=queries.device)
        mask_offdiag[row_indices, local_indices] = False

        sim_qd_pos = sim_qd[:, :total_size]
        mask_qd_pos = mask_offdiag & (sim_qd_pos <= threshold)
        qd_pos_logits = torch.where(mask_qd_pos, sim_qd_pos * self.scale, -torch.inf)

        hard_logits = None
        if docs_neg:
            sim_qd_neg = sim_qd[:, total_size:]
            if self.hard_negative_margin > 0:
                mask_qneg = sim_qd_neg <= (pos_scores[:, None] + self.hard_negative_margin)
                hard_logits = torch.where(mask_qneg, sim_qd_neg * self.scale, -torch.inf)
            else:
                hard_logits = sim_qd_neg * self.scale

        sim_qq = self.similarity_fct(local_queries, queries)
        mask_qq = mask_offdiag & (sim_qq <= threshold)
        qq_logits = torch.where(mask_qq, sim_qq * self.scale, -torch.inf)

        sim_dd = self.similarity_fct(local_docs, docs_pos)
        mask_dd = mask_offdiag & (sim_dd <= threshold)
        dd_logits = torch.where(mask_dd, sim_dd * self.scale, -torch.inf)

        pos_logits = pos_scores * self.scale

        candidates = [pos_logits[:, None], qq_logits, dd_logits, qd_pos_logits]
        if hard_logits is not None:
            candidates.insert(1, hard_logits)
        all_logits = torch.cat([cand.reshape(batch_size, -1) for cand in candidates], dim=1)
        log_z = torch.logsumexp(all_logits, dim=1)
        loss = -(pos_logits - log_z).mean()
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "similarity_fct": self.similarity_fct.__name__,
            "margin": self.margin,
            "hard_negative_margin": self.hard_negative_margin,
            "gather_across_devices": self.gather_across_devices,
        }

    @property
    def scale(self) -> float:
        return 1.0 / self.temperature
