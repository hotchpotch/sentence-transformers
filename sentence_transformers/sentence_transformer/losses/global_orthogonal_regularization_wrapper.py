from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.losses.cached_gist_embed import CachedGISTEmbedLoss
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_symmetric_ranking import (
    CachedMultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.sentence_transformer.losses.global_orthogonal_regularization import (
    GlobalOrthogonalRegularizationLoss,
)
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import MultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.model import SentenceTransformer


class ForwardDecorator:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.embeddings: list[Tensor] = []

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        output = self.fn(features)
        if "sentence_embedding" in output:
            self.embeddings.append(output["sentence_embedding"])
        return output


class CachedGORLossDecorator:
    def __init__(
        self,
        fn,
        gor_loss: GlobalOrthogonalRegularizationLoss,
        gor_weight: float,
    ) -> None:
        self.fn = fn
        self.gor_loss = gor_loss
        self.gor_weight = gor_weight
        self.last_loss_components: dict[str, Tensor] = {}

    def __call__(self, reps: list[list[Tensor]], *args, **kwargs) -> Tensor:
        base_loss = self.fn(reps, *args, **kwargs)
        gor_loss = base_loss.new_zeros(()) if self.gor_weight == 0 else self.gor_weight * self._compute_gor_loss(reps)
        with_backward = kwargs.get("with_backward", args[-1] if args and isinstance(args[-1], bool) else False)

        if with_backward:
            gor_loss.backward()
            total_loss = base_loss + gor_loss.detach()
        else:
            total_loss = base_loss + gor_loss

        self.last_loss_components = {
            "base_loss": base_loss.detach(),
            "gor": gor_loss.detach(),
        }
        return total_loss

    def _compute_gor_loss(self, reps: list[list[Tensor]]) -> Tensor:
        embeddings = [torch.cat(reps_per_column) for reps_per_column in reps]
        return sum(self.gor_loss.compute_loss_from_embeddings(embeddings).values())


class GlobalOrthogonalRegularizationWrapperLoss(nn.Module):
    cached_losses = (
        CachedMultipleNegativesRankingLoss,
        CachedGISTEmbedLoss,
        CachedMultipleNegativesSymmetricRankingLoss,
    )

    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        gor_weight: float = 1.0,
        gor_loss: GlobalOrthogonalRegularizationLoss | None = None,
    ) -> None:
        """
        Add Gemma-style Global Orthogonal Regularization to another Sentence Transformer loss.

        For regular losses, this wrapper captures the embeddings produced by the wrapped loss via a temporary
        ``model.forward`` decorator. For cached losses, the regularization is computed on the same cached embeddings
        used by the wrapped loss, so no additional model forward pass is required.
        """
        super().__init__()
        self.model = model
        self.loss = loss
        self.gor_weight = gor_weight
        self.gor_loss = gor_loss or GlobalOrthogonalRegularizationLoss(model)
        self.last_loss_components: dict[str, Tensor] = {}
        self._cached_gor_decorator: CachedGORLossDecorator | None = None

        if isinstance(loss, self.cached_losses):
            self._cached_gor_decorator = CachedGORLossDecorator(
                loss.calculate_loss,
                gor_loss=self.gor_loss,
                gor_weight=self.gor_weight,
            )
            loss.calculate_loss = self._cached_gor_decorator

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> Tensor | dict[str, Tensor]:
        if isinstance(self.loss, self.cached_losses):
            loss = self.loss(sentence_features, labels)
            if self._cached_gor_decorator is not None:
                self.last_loss_components = self._cached_gor_decorator.last_loss_components
            return loss

        original_forward = self.model.forward
        decorated_forward = ForwardDecorator(original_forward)
        try:
            self.model.forward = decorated_forward
            base_loss = self.loss(sentence_features, labels)
        finally:
            self.model.forward = original_forward

        base_loss_tensor = self._sum_loss(base_loss)
        if not decorated_forward.embeddings:
            raise ValueError(
                "GlobalOrthogonalRegularizationWrapperLoss requires the wrapped loss to call the model and produce "
                "'sentence_embedding'."
            )
        gor_loss = (
            base_loss_tensor.new_zeros(())
            if self.gor_weight == 0
            else self.gor_weight * sum(self.gor_loss.compute_loss_from_embeddings(decorated_forward.embeddings).values())
        )
        self.last_loss_components = {
            "base_loss": base_loss_tensor.detach(),
            "gor": gor_loss.detach(),
        }

        if isinstance(base_loss, dict):
            return {**base_loss, "gor": gor_loss}
        return base_loss + gor_loss

    @staticmethod
    def _sum_loss(loss: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(loss, dict):
            return torch.stack(list(loss.values())).sum()
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "gor_weight": self.gor_weight,
            "gor_loss": self.gor_loss.get_config_dict(),
        }


class InfoNCEWithGlobalOrthogonalRegularizationLoss(GlobalOrthogonalRegularizationWrapperLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        base_loss: MultipleNegativesRankingLoss | CachedMultipleNegativesRankingLoss,
        gor_weight: float = 1.0,
        gor_loss: GlobalOrthogonalRegularizationLoss | None = None,
    ) -> None:
        """
        Backwards-compatible alias for wrapping MultipleNegativesRankingLoss with GOR.

        Prefer :class:`GlobalOrthogonalRegularizationWrapperLoss` for new code, as the wrapper is not specific to
        InfoNCE.
        """
        super().__init__(model=model, loss=base_loss, gor_weight=gor_weight, gor_loss=gor_loss)
        self.base_loss = self.loss
