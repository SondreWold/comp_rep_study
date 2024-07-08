import pytest
import torch
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import (
    EOS_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    CollateFunctor,
    SequenceDataset,
)


@pytest.fixture
def test_data():
    test_path = "../data/base_tasks/scan_train.csv"
    return SequenceDataset(path=test_path)


def test_padding(test_data):
    test_size = 16
    test_items = [test_data[i] for i in range(0, 16)]
    lengths = [source_sequence.size(0) for source_sequence, _, _, _ in test_items]
    test_loader = DataLoader(
        test_items, batch_size=test_size, collate_fn=CollateFunctor(), shuffle=False
    )
    for source_ids, source_mask, target_ids, target_mask, x, y in test_loader:
        assert source_ids.size(1) == max(lengths), f"Failed at {source_ids}"
        for end_token in source_ids[:, -1]:
            assert end_token.item() in [
                PAD_TOKEN,
                EOS_TOKEN,
            ], "Ends on non special token"
        x_pads = torch.where(source_ids == PAD_TOKEN, 1.0, 0.0)
        actual_pads = torch.where(source_mask == bool(PAD_TOKEN), 0.0, 1.0)
        assert torch.all(
            torch.eq(x_pads, actual_pads)
        ), "Mask does match actual padding"
