from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, ConcatDataset, SequentialSampler

from sentence_transformers.sampler import NoDuplicatesBatchSampler, SmoothedProportionalBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def _make_dataset(start: int, length: int) -> Dataset:
    values = list(range(start, start + length))
    labels = [value % 2 for value in values]
    return Dataset.from_dict({"data": values, "label": labels})


@pytest.fixture
def dummy_concat_dataset() -> ConcatDataset:
    dataset_1 = _make_dataset(0, 12)
    dataset_2 = _make_dataset(100, 8)
    return ConcatDataset([dataset_1, dataset_2])


@pytest.fixture
def dummy_duplicates_dataset() -> Dataset:
    values = [{"anchor": "anchor_1", "positive": "positive_1"}] * 10 + [
        {"anchor": "anchor_2", "positive": "positive_2"}
    ] * 8
    return Dataset.from_list(values)


def test_smoothed_proportional_batch_sampler_len(dummy_concat_dataset: ConcatDataset) -> None:
    batch_size = 2
    batch_sampler_1 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[0]))), batch_size=batch_size, drop_last=True
    )
    batch_sampler_2 = BatchSampler(
        SequentialSampler(range(len(dummy_concat_dataset.datasets[1]))), batch_size=batch_size, drop_last=True
    )

    sampler = SmoothedProportionalBatchSampler(
        dataset=dummy_concat_dataset,
        batch_samplers=[batch_sampler_1, batch_sampler_2],
        generator=torch.Generator(),
        seed=7,
        smoothing=0.5,
    )

    base_batches = len(batch_sampler_1) + len(batch_sampler_2)
    assert len(sampler) == base_batches

    batches = list(iter(sampler))
    assert len(batches) == len(sampler)

    offset = len(dummy_concat_dataset.datasets[0])
    for batch in batches:
        assert all(idx < offset for idx in batch) or all(idx >= offset for idx in batch)


def test_smoothed_proportional_batch_sampler_skips_empty_sampler() -> None:
    dataset_1 = _make_dataset(0, 2)
    dataset_2 = _make_dataset(100, 8)
    concat_dataset = ConcatDataset([dataset_1, dataset_2])

    batch_size = 4
    batch_sampler_1 = BatchSampler(SequentialSampler(range(len(dataset_1))), batch_size=batch_size, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(len(dataset_2))), batch_size=batch_size, drop_last=True)

    sampler = SmoothedProportionalBatchSampler(
        dataset=concat_dataset,
        batch_samplers=[batch_sampler_1, batch_sampler_2],
        generator=torch.Generator(),
        seed=5,
        smoothing=0.5,
    )

    batches = list(iter(sampler))
    assert len(batches) == len(sampler)
    assert len(sampler) == len(batch_sampler_2)

    offset = len(dataset_1)
    assert all(all(idx >= offset for idx in batch) for batch in batches)


def test_smoothed_proportional_batch_sampler_no_duplicates(dummy_duplicates_dataset: Dataset) -> None:
    batch_size = 2
    sampler_1 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["anchor"]
    )
    sampler_2 = NoDuplicatesBatchSampler(
        dataset=dummy_duplicates_dataset, batch_size=batch_size, drop_last=True, valid_label_columns=["positive"]
    )

    concat_dataset = ConcatDataset([dummy_duplicates_dataset, dummy_duplicates_dataset])
    sampler = SmoothedProportionalBatchSampler(
        dataset=concat_dataset,
        batch_samplers=[sampler_1, sampler_2],
        generator=torch.Generator(),
        seed=13,
        smoothing=0.5,
    )

    batches = list(iter(sampler))
    offset = len(dummy_duplicates_dataset)

    for batch in batches:
        assert all(idx < offset for idx in batch) or all(idx >= offset for idx in batch)
        if all(idx < offset for idx in batch):
            values = [concat_dataset[idx]["positive"] for idx in batch]
        else:
            values = [concat_dataset[idx]["anchor"] for idx in batch]
        assert len(values) == len(set(values))


def test_smoothed_proportional_batch_sampler_weights() -> None:
    dataset_1 = _make_dataset(0, 80)
    dataset_2 = _make_dataset(100, 20)
    concat_dataset = ConcatDataset([dataset_1, dataset_2])

    batch_size = 4
    batch_sampler_1 = BatchSampler(SequentialSampler(range(len(dataset_1))), batch_size=batch_size, drop_last=True)
    batch_sampler_2 = BatchSampler(SequentialSampler(range(len(dataset_2))), batch_size=batch_size, drop_last=True)

    sampler = SmoothedProportionalBatchSampler(
        dataset=concat_dataset,
        batch_samplers=[batch_sampler_1, batch_sampler_2],
        generator=torch.Generator(),
        seed=3,
        smoothing=0.5,
    )
    weights = sampler._compute_weights([0, 1])
    expected = torch.tensor([len(dataset_1), len(dataset_2)], dtype=torch.float64).pow(0.5)
    expected = expected / expected.sum()
    assert torch.allclose(weights, expected)
