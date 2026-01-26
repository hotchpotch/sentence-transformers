from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesMaskedBidirectionalRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # TODO: This if-statement is for if the model does not require gradients, which may happen if the model
                # contains a Router where one of the routes is frozen. It should be possible to not have to call
                # embed_minibatch_iter in that case, as it's unnecessarily expensive.
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedMultipleNegativesMaskedBidirectionalRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        temperature: float = 0.01,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        margin: float | None = 0.1,
        hard_negative_margin: float | None = None,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Cached variant of MultipleNegativesMaskedBidirectionalRankingLoss using GradCache.

        This masked InfoNCE variant uses three pools in the denominator: q->d, q->q, and d->d. It applies a
        margin-based mask to suppress likely false negatives and excludes hard negatives from the d->d term.

        Args:
            model: SentenceTransformer model
            temperature: Temperature parameter to scale the similarities. The internal scale is derived as
                ``scale = 1 / temperature``, so temperature=0.01 is equivalent to scale=100.0.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            margin: Margin for masking in-batch candidates. Candidates with similarity > s_pos + margin are masked.
                If None, in-batch masking is disabled (except for diagonal exclusion).
            hard_negative_margin: Margin for masking hard negatives. If None, hard negative filtering is disabled.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets
            2. Optional negatives are supported as hard negatives (additional documents).
            3. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance,
               but slower training time than MultipleNegativesMaskedBidirectionalRankingLoss.

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
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedMultipleNegativesMaskedBidirectionalRankingLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using MultipleNegativesMaskedBidirectionalRankingLoss instead."
            )

        self.model = model
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = temperature
        self.similarity_fct = similarity_fct
        self.mini_batch_size = mini_batch_size
        self.margin = margin
        self.hard_negative_margin = hard_negative_margin
        self.gather_across_devices = gather_across_devices
        self.show_progress_bar = show_progress_bar

        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of sentences."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches of sentences for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        batch_size, _ = input_ids.shape
        for i, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the loss and cache gradients."""
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], with_backward: bool = False) -> Tensor:
        """Calculate the masked InfoNCE loss without caching gradients (for evaluation)."""
        if len(reps) < 2:
            raise ValueError(f"Expected at least 2 embeddings, got {len(reps)}")

        queries = torch.cat(reps[0])
        docs_pos = torch.cat(reps[1])
        docs_neg = [torch.cat(r) for r in reps[2:]]
        batch_size = len(queries)
        offset = 0

        if self.gather_across_devices:
            queries = all_gather_with_grad(queries)
            docs_pos = all_gather_with_grad(docs_pos)
            docs_neg = [all_gather_with_grad(doc) for doc in docs_neg]
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        total_size = len(queries)
        docs_all = torch.cat([docs_pos] + docs_neg, dim=0) if docs_neg else docs_pos
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)

        losses: list[torch.Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = min(begin + self.mini_batch_size, batch_size)
            local_batch = local_indices[begin:end]
            local_queries = queries[local_batch]
            local_docs = docs_pos[local_batch]

            sim_qd = self.similarity_fct(local_queries, docs_all)
            row_indices = torch.arange(len(local_batch), device=queries.device)
            pos_scores = sim_qd[row_indices, local_batch]
            threshold = None if self.margin is None else pos_scores[:, None] + self.margin

            mask_offdiag = torch.ones(len(local_batch), total_size, dtype=torch.bool, device=queries.device)
            mask_offdiag[row_indices, local_batch] = False

            sim_qd_pos = sim_qd[:, :total_size]
            if threshold is None:
                mask_qd_pos = mask_offdiag
            else:
                mask_qd_pos = mask_offdiag & (sim_qd_pos <= threshold)
            qd_pos_logits = torch.where(mask_qd_pos, sim_qd_pos * self.scale, -torch.inf)

            hard_logits = None
            masked_hard = None
            total_hard = None
            if docs_neg:
                sim_qd_neg = sim_qd[:, total_size:]
                if self.hard_negative_margin is None:
                    hard_logits = sim_qd_neg * self.scale
                    total_hard = sim_qd_neg.numel()
                    masked_hard = 0
                else:
                    mask_qneg = sim_qd_neg <= (pos_scores[:, None] + self.hard_negative_margin)
                    hard_logits = torch.where(mask_qneg, sim_qd_neg * self.scale, -torch.inf)
                    total_hard = sim_qd_neg.numel()
                    masked_hard = total_hard - mask_qneg.sum().item()

            sim_qq = self.similarity_fct(local_queries, queries)
            if threshold is None:
                mask_qq = mask_offdiag
            else:
                mask_qq = mask_offdiag & (sim_qq <= threshold)
            qq_logits = torch.where(mask_qq, sim_qq * self.scale, -torch.inf)

            sim_dd = self.similarity_fct(local_docs, docs_pos)
            if threshold is None:
                mask_dd = mask_offdiag
            else:
                mask_dd = mask_offdiag & (sim_dd <= threshold)
            dd_logits = torch.where(mask_dd, sim_dd * self.scale, -torch.inf)

            pos_logits = pos_scores * self.scale

            candidates = [pos_logits[:, None], qq_logits, dd_logits, qd_pos_logits]
            if hard_logits is not None:
                candidates.insert(1, hard_logits)
            all_logits = torch.cat([cand.reshape(len(local_batch), -1) for cand in candidates], dim=1)
            log_z = torch.logsumexp(all_logits, dim=1)

            loss_mbatch = -(pos_logits - log_z).mean()
            loss_mbatch = loss_mbatch * len(local_batch) / batch_size
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        return sum(losses)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "similarity_fct": self.similarity_fct.__name__,
            "mini_batch_size": self.mini_batch_size,
            "margin": self.margin,
            "hard_negative_margin": self.hard_negative_margin,
            "gather_across_devices": self.gather_across_devices,
        }

    @property
    def scale(self) -> float:
        return 1.0 / self.temperature
