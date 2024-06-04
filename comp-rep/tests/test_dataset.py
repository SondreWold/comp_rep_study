from comp_rep.dataset import (
    SequenceDataset,
    SOS_TOKEN,
    PAD_TOKEN,
    EOS_TOKEN,
    CollateFunctor,
)
from torch.utils.data import DataLoader
import pytest


@pytest.fixture
def test_data():
    test_path = "../data/base_tasks/scan_train.csv"
    return SequenceDataset(path=test_path)


def test_getitem_sos_eos(test_data):
    input_tensor, output_tensor, x, y = test_data[0]
    assert output_tensor[0].item() == SOS_TOKEN
    assert output_tensor[-1].item() == EOS_TOKEN


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
            assert end_token.item() in [PAD_TOKEN, EOS_TOKEN]
