from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import InputExample, SentenceTransformer, losses, util


class _DummyModel:
    def __getitem__(self, idx):
        return object()


@pytest.fixture(scope="module")
def shared_sbert() -> SentenceTransformer:
    # Reuse a single model instance to avoid repeated downloads and initialization in CI.
    model = SentenceTransformer("distilbert-base-uncased")
    model.to("cpu")
    return model


def _manual_masked_bidirectional_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: list[torch.Tensor],
    margin: float | None,
    hard_negative_margin: float | None,
    scale: float = 1.0,
) -> torch.Tensor:
    docs_all = torch.cat([positives] + negatives, dim=0) if negatives else positives
    sim_qd = util.dot_score(queries, docs_all)
    sim_qq = util.dot_score(queries, queries)
    sim_dd = util.dot_score(positives, positives)

    batch_size = queries.size(0)
    row_indices = torch.arange(batch_size)
    pos_scores = sim_qd[row_indices, row_indices]
    threshold = None if margin is None else pos_scores[:, None] + margin

    mask_offdiag = torch.ones(batch_size, batch_size, dtype=torch.bool)
    mask_offdiag[row_indices, row_indices] = False

    sim_qd_pos = sim_qd[:, :batch_size]
    if threshold is None:
        mask_qd_pos = mask_offdiag
    else:
        mask_qd_pos = mask_offdiag & (sim_qd_pos <= threshold)
    qd_pos_logits = torch.where(mask_qd_pos, sim_qd_pos * scale, -torch.inf)

    hard_logits = None
    if negatives:
        sim_qd_neg = sim_qd[:, batch_size:]
        if hard_negative_margin is None:
            hard_logits = sim_qd_neg * scale
        else:
            mask_qneg = sim_qd_neg <= (pos_scores[:, None] + hard_negative_margin)
            hard_logits = torch.where(mask_qneg, sim_qd_neg * scale, -torch.inf)

    if threshold is None:
        mask_qq = mask_offdiag
    else:
        mask_qq = mask_offdiag & (sim_qq <= threshold)
    qq_logits = torch.where(mask_qq, sim_qq * scale, -torch.inf)

    if threshold is None:
        mask_dd = mask_offdiag
    else:
        mask_dd = mask_offdiag & (sim_dd <= threshold)
    dd_logits = torch.where(mask_dd, sim_dd * scale, -torch.inf)

    pos_logits = pos_scores * scale

    candidates = [pos_logits[:, None], qq_logits, dd_logits, qd_pos_logits]
    if hard_logits is not None:
        candidates.insert(1, hard_logits)
    all_logits = torch.cat([cand.reshape(batch_size, -1) for cand in candidates], dim=1)
    log_z = torch.logsumexp(all_logits, dim=1)
    loss = -(pos_logits - log_z).mean()
    return loss


def test_masked_bidirectional_manual_formula_with_hard_negatives():
    loss = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=0.1,
        hard_negative_margin=0.0,
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [1.3, 0.2]])
    negatives = torch.tensor([[0.5, 0.0], [0.0, 0.5]])

    computed = loss.compute_loss_from_embeddings([queries, positives, negatives], labels=None)
    expected = _manual_masked_bidirectional_loss(queries, positives, [negatives], 0.1, 0.0, scale=1.0)

    assert pytest.approx(computed.item(), rel=1e-6) == expected.item()


def test_masked_bidirectional_hard_negative_margin_filters():
    loss = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=0.2,
        hard_negative_margin=0.1,
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negatives = torch.tensor([[1.2, 0.0], [0.0, 1.2]])

    computed = loss.compute_loss_from_embeddings([queries, positives, negatives], labels=None)
    expected = _manual_masked_bidirectional_loss(queries, positives, [negatives], 0.2, 0.1, scale=1.0)

    assert pytest.approx(computed.item(), rel=1e-6) == expected.item()

    loss_no_filter = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=0.2,
        hard_negative_margin=None,
    )
    computed_no_filter = loss_no_filter.compute_loss_from_embeddings([queries, positives, negatives], labels=None)

    assert computed.item() != pytest.approx(computed_no_filter.item(), rel=1e-6)


def test_masked_bidirectional_margin_none_skips_filtering():
    loss = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=None,
        hard_negative_margin=None,
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negatives = torch.tensor([[1.5, 0.0], [0.0, 1.5]])

    computed = loss.compute_loss_from_embeddings([queries, positives, negatives], labels=None)
    expected = _manual_masked_bidirectional_loss(queries, positives, [negatives], None, None, scale=1.0)

    assert pytest.approx(computed.item(), rel=1e-6) == expected.item()


def test_masked_bidirectional_margin_masks_high_similarity_candidates():
    # Ensure candidates above s_pos + margin are masked out.
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[0.5, 0.0], [2.0, 0.0]])

    loss_masked = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=0.1,
        hard_negative_margin=None,
    )
    loss_unmasked = losses.MultipleNegativesMaskedBidirectionalRankingLoss(
        model=Mock(spec=SentenceTransformer),
        temperature=1.0,
        similarity_fct=util.dot_score,
        margin=None,
        hard_negative_margin=None,
    )

    masked_value = loss_masked.compute_loss_from_embeddings([queries, positives], labels=None).item()
    unmasked_value = loss_unmasked.compute_loss_from_embeddings([queries, positives], labels=None).item()

    assert masked_value < unmasked_value


@pytest.mark.parametrize(
    ["train_samples", "scaler", "precision"],
    [
        (
            [
                InputExample(texts=[q, p])
                for q, p in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                )
            ],
            1.0,
            1e-4,
        ),
    ],
)
def test_cached_masked_bidirectional_info_nce_same_grad(
    train_samples: list[InputExample],
    scaler: float,
    precision: float,
    shared_sbert: SentenceTransformer,
):
    optimizer = Adam(shared_sbert.parameters())

    set_seed(42)
    optimizer.zero_grad()
    loss_base = losses.MultipleNegativesMaskedBidirectionalRankingLoss(shared_sbert)
    loss_base_value: torch.Tensor = loss_base.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_base_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_base.named_parameters() if p.grad is not None}

    set_seed(42)
    optimizer.zero_grad()
    loss_cached = losses.CachedMultipleNegativesMaskedBidirectionalRankingLoss(shared_sbert, mini_batch_size=2)
    loss_cached_value: torch.Tensor = loss_cached.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_cached_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cached.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_base_value.item(), rel=precision) == loss_cached_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    assert nclose == len(grad_expected)


def test_cached_masked_bidirectional_info_nce_same_grad_with_hard_negatives(shared_sbert: SentenceTransformer):
    train_samples = [
        InputExample(texts=[q, p, n])
        for q, p, n in zip(
            ["aaa", "bbb", "ccc", "ddd"],
            ["aas", "bbs", "ccs", "dds"],
            ["zzz", "yyy", "xxx", "www"],
        )
    ]
    scaler = 1.0
    precision = 1e-4
    optimizer = Adam(shared_sbert.parameters())

    set_seed(42)
    optimizer.zero_grad()
    loss_base = losses.MultipleNegativesMaskedBidirectionalRankingLoss(shared_sbert)
    loss_base_value: torch.Tensor = loss_base.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_base_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_base.named_parameters() if p.grad is not None}

    set_seed(42)
    optimizer.zero_grad()
    loss_cached = losses.CachedMultipleNegativesMaskedBidirectionalRankingLoss(shared_sbert, mini_batch_size=2)
    loss_cached_value: torch.Tensor = loss_cached.forward(*shared_sbert.smart_batching_collate(train_samples)) * scaler
    loss_cached_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cached.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_base_value.item(), rel=precision) == loss_cached_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    assert nclose == len(grad_expected)
