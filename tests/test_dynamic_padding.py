import torch

from training.train_dense_openr1 import (
    CausalBatchCollator,
    LengthGroupedBatchSampler,
    PackedCausalDataset,
    TokenSequence,
)


def test_dynamic_collator_right_pads_and_masks_labels():
    ds = PackedCausalDataset(
        [
            TokenSequence(ids=[10, 11, 12, 2], label_mask=[0, 1, 1, 1]),
            TokenSequence(ids=[20, 21, 2], label_mask=[0, 1, 1]),
        ],
        seq_len=8,
        pad_token_id=0,
        pack_samples=False,
    )
    collate = CausalBatchCollator(
        pad_token_id=0,
        max_seq_len=8,
        pad_to_multiple=4,
        static_padding=False,
    )

    batch = collate([ds[0], ds[1]])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["seq_lens"].tolist() == [3, 2]
    assert batch["input_ids"][1].tolist() == [20, 21, 0, 0]
    assert batch["labels"][1].tolist() == [21, 2, -100, -100]
    assert torch.equal(batch["labels"][0], torch.tensor([11, 12, 2, -100]))


def test_eos_label_is_preserved_when_pad_equals_eos():
    ds = PackedCausalDataset(
        [
            TokenSequence(ids=[10, 11, 2], label_mask=[0, 1, 1]),
            TokenSequence(ids=[20, 21, 22, 2], label_mask=[0, 1, 1, 1]),
        ],
        seq_len=8,
        pad_token_id=2,
        pack_samples=False,
    )
    collate = CausalBatchCollator(
        pad_token_id=2,
        max_seq_len=8,
        pad_to_multiple=4,
        static_padding=False,
    )

    batch = collate([ds[0], ds[1]])

    assert batch["input_ids"][0].tolist() == [10, 11, 2, 2]
    assert batch["labels"][0].tolist() == [11, 2, -100, -100]
    assert batch["labels"][1].tolist() == [21, 22, 2, -100]


def test_dynamic_collator_caps_alignment_at_max_seq_len():
    ds = PackedCausalDataset(
        [TokenSequence(ids=list(range(10)), label_mask=[1] * 10)],
        seq_len=9,
        pad_token_id=0,
        pack_samples=False,
    )
    collate = CausalBatchCollator(
        pad_token_id=0,
        max_seq_len=9,
        pad_to_multiple=8,
        static_padding=False,
    )

    batch = collate([ds[0]])

    assert batch["input_ids"].shape == (1, 9)
    assert batch["seq_lens"].item() == 9


def test_length_grouped_sampler_is_shuffled_but_local_by_length():
    lengths = [8, 9, 10, 11, 90, 91, 92, 93]
    sampler = LengthGroupedBatchSampler(
        lengths=lengths,
        batch_size=2,
        drop_last=True,
        seed=123,
        bucket_mult=4,
    )

    batches = list(iter(sampler))
    flat = sorted(idx for batch in batches for idx in batch)

    assert flat == list(range(len(lengths)))
    assert all(len(batch) == 2 for batch in batches)
    assert all(abs(lengths[a] - lengths[b]) <= 1 for a, b in batches)
