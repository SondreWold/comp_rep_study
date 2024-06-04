from comp_rep.dataset import SequenceDataset, SOS_TOKEN, PAD_TOKEN, EOS_TOKEN

def test_getitem_sos_eos():
    test_path = "../data/base_tasks/scan_train.csv"
    dataset = SequenceDataset(path=test_path)
    input_tensor, output_tensor, x, y = dataset[0]
    assert output_tensor[0].item() == SOS_TOKEN
    assert output_tensor[-1].item() == EOS_TOKEN
