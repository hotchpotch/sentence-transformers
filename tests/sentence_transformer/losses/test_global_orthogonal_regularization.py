from __future__ import annotations

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    GlobalOrthogonalRegularizationLoss,
    GlobalOrthogonalRegularizationWrapperLoss,
    InfoNCEWithGlobalOrthogonalRegularizationLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)


class _FakeModel(torch.nn.Module):
    def __getitem__(self, index: int) -> torch.nn.Module:
        return torch.nn.Linear(1, 1)

    def forward(self, features):
        return {"sentence_embedding": features["sentence_embedding"]}

    def get_embedding_dimension(self):
        return None


class _GenericEmbeddingLoss(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels=None):
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return sum(embedding.pow(2).mean() for embedding in embeddings)


def test_gemma_gor_second_moment_only_defaults() -> None:
    loss = GlobalOrthogonalRegularizationLoss(model=None)

    embeddings = [
        torch.eye(3),
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    ]
    output = loss.compute_loss_from_embeddings(embeddings)

    assert set(output) == {"gor_second_moment"}
    assert output["gor_second_moment"] == pytest.approx(1.0)


def test_gor_original_dimension_threshold_remains_available() -> None:
    loss = GlobalOrthogonalRegularizationLoss(
        model=None,
        aggregation="mean",
        second_moment_threshold="dimension",
    )
    embeddings = [torch.eye(3)]

    output = loss.compute_loss_from_embeddings(embeddings)

    assert output["gor_second_moment"] == pytest.approx(0.0)


def test_gor_requires_at_least_two_embeddings_per_column() -> None:
    loss = GlobalOrthogonalRegularizationLoss(model=None)

    with pytest.raises(ValueError, match="requires at least 2 embeddings"):
        loss.compute_loss_from_embeddings([torch.ones(1, 2)])


def test_infonce_gor_matches_manual_non_cached_composition() -> None:
    model = _FakeModel()
    base_loss = MultipleNegativesRankingLoss(model)
    gor_loss = GlobalOrthogonalRegularizationLoss(model)
    wrapper = InfoNCEWithGlobalOrthogonalRegularizationLoss(model, base_loss, gor_weight=0.3, gor_loss=gor_loss)
    labels = torch.zeros(3, dtype=torch.long)
    features = [
        {
            "sentence_embedding": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                requires_grad=True,
            )
        },
        {
            "sentence_embedding": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]],
                requires_grad=True,
            )
        },
    ]
    embeddings = [feature["sentence_embedding"] for feature in features]

    actual = wrapper(features, labels)
    expected = base_loss.compute_loss_from_embeddings(embeddings, labels)
    expected = expected + 0.3 * sum(gor_loss.compute_loss_from_embeddings(embeddings).values())

    assert actual.detach().item() == pytest.approx(expected.detach().item())
    assert wrapper.last_loss_components["base_loss"] == pytest.approx(
        base_loss.compute_loss_from_embeddings(embeddings, labels).detach().item()
    )


def test_gor_wrapper_supports_generic_non_cached_loss() -> None:
    model = _FakeModel()
    base_loss = _GenericEmbeddingLoss(model)
    gor_loss = GlobalOrthogonalRegularizationLoss(model)
    wrapper = GlobalOrthogonalRegularizationWrapperLoss(model, base_loss, gor_weight=0.3, gor_loss=gor_loss)
    features = [
        {"sentence_embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)},
        {"sentence_embedding": torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)},
    ]
    embeddings = [feature["sentence_embedding"] for feature in features]

    actual = wrapper(features, labels=None)
    expected = base_loss(features) + 0.3 * sum(gor_loss.compute_loss_from_embeddings(embeddings).values())

    assert actual.detach().item() == pytest.approx(expected.detach().item())
    assert wrapper.last_loss_components["base_loss"] == pytest.approx(base_loss(features).detach().item())


def test_cached_infonce_gor_gradients_match_manual_composition() -> None:
    model = _FakeModel()
    labels = torch.zeros(3, dtype=torch.long)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)
    docs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], requires_grad=True)
    manual_base_loss = MultipleNegativesRankingLoss(model)
    gor_loss = GlobalOrthogonalRegularizationLoss(model)
    manual_loss = manual_base_loss.compute_loss_from_embeddings([queries, docs], labels)
    manual_loss = manual_loss + 0.3 * sum(gor_loss.compute_loss_from_embeddings([queries, docs]).values())
    manual_loss.backward()

    cached_queries = queries.detach().clone().requires_grad_()
    cached_docs = docs.detach().clone().requires_grad_()
    cached_base_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3)
    InfoNCEWithGlobalOrthogonalRegularizationLoss(
        model,
        cached_base_loss,
        gor_weight=0.3,
        gor_loss=GlobalOrthogonalRegularizationLoss(model),
    )

    cached_loss = cached_base_loss.calculate_loss([[cached_queries], [cached_docs]], with_backward=True)

    assert cached_loss.item() == pytest.approx(manual_loss.detach().item())
    assert torch.allclose(cached_queries.grad, queries.grad)
    assert torch.allclose(cached_docs.grad, docs.grad)


def test_cached_infonce_gor_decorator_can_stack_with_matryoshka() -> None:
    model = _FakeModel()
    cached_base_loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=3)
    wrapper = InfoNCEWithGlobalOrthogonalRegularizationLoss(model, cached_base_loss)

    MatryoshkaLoss(model, wrapper.base_loss, matryoshka_dims=[2])
    reps = [
        [torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True)],
        [torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], requires_grad=True)],
    ]

    loss = cached_base_loss.calculate_loss(reps, with_backward=True)

    assert loss.ndim == 0
    assert all(rep.grad is not None for reps_per_column in reps for rep in reps_per_column)
