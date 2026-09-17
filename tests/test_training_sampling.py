from __future__ import annotations

from scripts.train_small import sample_validation_batches


def test_validation_sampling_is_fixed_random_and_non_contiguous() -> None:
    first = sample_validation_batches(
        dataset_size=1000,
        batch_size=4,
        max_batches=10,
        seed=123,
    )
    second = sample_validation_batches(
        dataset_size=1000,
        batch_size=4,
        max_batches=10,
        seed=123,
    )
    different_seed = sample_validation_batches(
        dataset_size=1000,
        batch_size=4,
        max_batches=10,
        seed=456,
    )

    first_indices = [index for batch in first for index in batch]
    assert first == second
    assert first != different_seed
    assert len(first_indices) == len(set(first_indices))
    assert min(first_indices) >= 0
    assert max(first_indices) < 1000
    assert first_indices != list(range(40))
