from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.losses.cached_gist_embed import CachedGISTEmbedLoss
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    CachedMultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_symmetric_ranking import (
    CachedMultipleNegativesSymmetricRankingLoss,
)
from sentence_transformers.sentence_transformer.model import SentenceTransformer


def quantize_embeddings_torch(
    embeddings: Tensor, precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
) -> Tensor:
    if precision == "float32":
        return embeddings

    if precision in ("int8", "uint8"):
        mins = embeddings.min(dim=0, keepdim=True).values
        maxs = embeddings.max(dim=0, keepdim=True).values
        steps = torch.clamp((maxs - mins) / 255, min=1e-8)
        normalized = (embeddings - mins) / steps
        quantized = normalized + (torch.round(normalized) - normalized).detach()

        if precision == "int8":
            quantized = torch.clamp(quantized - 128, -128, 127)
            return (quantized + 128) * steps + mins

        quantized = torch.clamp(quantized, 0, 255)
        return quantized * steps + mins

    if precision in ("binary", "ubinary"):
        quantized = (embeddings > 0).to(embeddings.dtype)
        return embeddings + (quantized - embeddings).detach()

    raise ValueError(f"Unsupported precision: {precision}")


class ForwardDecorator:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.precision: str | None = None
        self.cache: list[dict[str, Tensor]] = []
        self.caching_mode = True
        self.idx = 0

    def set_precision(self, precision: str | None) -> None:
        self.precision = precision
        self.idx = 0

    def start_caching(self) -> None:
        self.caching_mode = True
        self.cache = []
        self.idx = 0

    def use_cache(self) -> None:
        self.caching_mode = False
        self.idx = 0

    def __call__(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.caching_mode:
            output = self.fn(features)
            self.cache.append(output)
        else:
            output = dict(self.cache[self.idx])

        if self.precision is not None:
            if "token_embeddings" in output:
                output["token_embeddings"] = quantize_embeddings_torch(output["token_embeddings"], self.precision)
            output["sentence_embedding"] = quantize_embeddings_torch(output["sentence_embedding"], self.precision)

        self.idx += 1
        return output


class CachedLossDecorator:
    def __init__(
        self,
        fn,
        quantization_precisions: Sequence[Literal["float32", "int8", "uint8", "binary", "ubinary"]],
        quantization_weights: Sequence[float] | Sequence[int],
        n_precisions_per_step: int = -1,
    ) -> None:
        self.fn = fn
        self.quantization_precisions = quantization_precisions
        self.quantization_weights = quantization_weights
        self.n_precisions_per_step = n_precisions_per_step

    def __call__(self, reps: list[list[Tensor]], *args, **kwargs) -> Tensor:
        precision_indices = range(len(self.quantization_precisions))
        if self.n_precisions_per_step > 0 and self.n_precisions_per_step < len(self.quantization_precisions):
            precision_indices = random.sample(list(precision_indices), self.n_precisions_per_step)

        loss = None
        with_backward = kwargs.get("with_backward", args[-1] if args and isinstance(args[-1], bool) else False)
        for idx in precision_indices:
            precision = self.quantization_precisions[idx]
            weight = self.quantization_weights[idx]
            transformed = [[quantize_embeddings_torch(r, precision) for r in minibatch] for minibatch in reps]

            if with_backward:
                loss_reps = [[r.detach().requires_grad_() for r in minibatch] for minibatch in transformed]
                step_loss = self.fn(loss_reps, *args, **kwargs)
                for transformed_minibatch, loss_minibatch in zip(transformed, loss_reps):
                    for transformed_embedding, loss_embedding in zip(transformed_minibatch, loss_minibatch):
                        if loss_embedding.grad is not None:
                            transformed_embedding.backward(weight * loss_embedding.grad)
                weighted_step_loss = weight * step_loss.detach()
            else:
                weighted_step_loss = weight * self.fn(transformed, *args, **kwargs)

            loss = weighted_step_loss if loss is None else loss + weighted_step_loss

        if loss is None:
            raise ValueError("No quantization precision was selected for this step.")
        return loss


class QuantizationAwareLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        loss: nn.Module,
        quantization_precisions: Sequence[Literal["float32", "int8", "uint8", "binary", "ubinary"]],
        quantization_weights: Sequence[float] | Sequence[int] | None = None,
        n_precisions_per_step: int = -1,
    ) -> None:
        """
        Loss modifier that trains a Sentence Transformer loss across multiple embedding quantization precisions.

        This is analogous to :class:`MatryoshkaLoss`: it wraps an existing loss, reuses the embeddings produced by the
        wrapped loss, applies differentiable quantization with straight-through estimators, and sums the weighted losses.
        Cached in-batch-negative losses are supported by decorating their ``calculate_loss`` method.
        """
        super().__init__()
        self.model = model
        self.loss = loss

        if not quantization_precisions:
            raise ValueError("You must provide at least one quantization precision in quantization_precisions.")

        valid_precisions = {"float32", "int8", "uint8", "binary", "ubinary"}
        for precision in quantization_precisions:
            if precision not in valid_precisions:
                raise ValueError(f"Invalid precision {precision!r}. Valid precisions are: {sorted(valid_precisions)}")

        if quantization_weights is None:
            quantization_weights = [1] * len(quantization_precisions)
        elif len(quantization_weights) != len(quantization_precisions):
            raise ValueError("quantization_weights must be the same length as quantization_precisions.")

        self.quantization_precisions = tuple(quantization_precisions)
        self.quantization_weights = tuple(quantization_weights)
        self.n_precisions_per_step = n_precisions_per_step
        self.cached_losses = (
            CachedMultipleNegativesRankingLoss,
            CachedGISTEmbedLoss,
            CachedMultipleNegativesSymmetricRankingLoss,
        )
        if isinstance(loss, self.cached_losses):
            loss.calculate_loss = CachedLossDecorator(
                loss.calculate_loss,
                self.quantization_precisions,
                self.quantization_weights,
                n_precisions_per_step,
            )

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None
    ) -> dict[str, Tensor] | Tensor:
        if isinstance(self.loss, self.cached_losses):
            return self.loss(sentence_features, labels)

        original_forward = self.model.forward
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            precision_indices = range(len(self.quantization_precisions))
            if self.n_precisions_per_step > 0 and self.n_precisions_per_step < len(self.quantization_precisions):
                precision_indices = random.sample(list(precision_indices), self.n_precisions_per_step)
                precision_indices.sort()

            losses = {}
            decorated_forward.start_caching()
            decorated_forward.set_precision(None)
            base_loss = self.loss(sentence_features, labels)

            if "float32" in self.quantization_precisions:
                float32_idx = self.quantization_precisions.index("float32")
                losses["qat_float32"] = self.quantization_weights[float32_idx] * base_loss

            decorated_forward.use_cache()
            for idx in precision_indices:
                precision = self.quantization_precisions[idx]
                if precision == "float32":
                    continue

                weight = self.quantization_weights[idx]
                decorated_forward.set_precision(precision)

                precision_labels = labels
                if isinstance(labels, Tensor) and labels.ndim >= 2:
                    precision_labels = quantize_embeddings_torch(labels, precision)

                losses[f"qat_{precision}"] = weight * self.loss(sentence_features, precision_labels)
        finally:
            self.model.forward = original_forward
        return losses

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss.__class__.__name__,
            "quantization_precisions": self.quantization_precisions,
            "quantization_weights": self.quantization_weights,
            "n_precisions_per_step": self.n_precisions_per_step,
        }

    @property
    def citation(self) -> str:
        return """
@article{jacob2018quantization,
    title={Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference},
    author={Jacob, Benoit and Kligys, Skirmantas and Chen, Bo and Zhu, Menglong and Tang, Matthew and Howard, Andrew and Kalenichenko, Dmitry},
    journal={arXiv preprint arXiv:1712.05877},
    year={2018}
}
"""
