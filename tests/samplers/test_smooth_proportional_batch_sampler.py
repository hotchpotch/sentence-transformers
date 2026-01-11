from __future__ import annotations

import pytest
import torch
from torch.utils.data import BatchSampler, ConcatDataset

from sentence_transformers.sampler import SmoothProportionalBatchSampler
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


class FixedBatchSampler(BatchSampler):
    def __init__(self, batches: list[list[int]]) -> None:
        self._batches = batches
        self.batch_size = len(batches[0]) if batches else 0
        self.drop_last = True

    def __iter__(self):
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


def test_smooth_proportional_batch_sampler_uses_batch_counts() -> None:
    dataset_1 = Dataset.from_dict({"data": [0, 1]})
    dataset_2 = Dataset.from_dict({"data": [0, 1]})
    concat_dataset = ConcatDataset([dataset_1, dataset_2])

    batches_1 = [list(range(2))] * 30
    batches_2 = [list(range(2))] * 70
    batch_sampler_1 = FixedBatchSampler(batches_1)
    batch_sampler_2 = FixedBatchSampler(batches_2)

    generator = torch.Generator()
    seed = 0
    sampler = SmoothProportionalBatchSampler(
        dataset=concat_dataset,
        batch_samplers=[batch_sampler_1, batch_sampler_2],
        alpha=0.5,
        generator=generator,
        seed=seed,
    )

    expected_generator = torch.Generator().manual_seed(seed)
    weights = torch.tensor([len(batch_sampler_1), len(batch_sampler_2)], dtype=torch.float).pow(0.5)
    expected_dataset_indices = torch.multinomial(
        weights, len(sampler), replacement=True, generator=expected_generator
    ).tolist()

    batches = list(iter(sampler))
    assert len(batches) == len(expected_dataset_indices)

    dataset_offset = len(dataset_1)
    actual_dataset_indices = [0 if batch[0] < dataset_offset else 1 for batch in batches]

    assert actual_dataset_indices == expected_dataset_indices
    assert actual_dataset_indices.count(0) > len(batch_sampler_1)
