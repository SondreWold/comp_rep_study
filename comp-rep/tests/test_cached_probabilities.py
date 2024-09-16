from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cache_activations import run_caching
from comp_rep.data_prep.dataset import (
    CollateFunctor,
    CollateFunctorWithProbabilities,
    SequenceDataset,
    SequenceDatasetWithProbabilities,
)
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.models.model import Transformer
from comp_rep.pruning.activation_pruning.activation_pruner import ActivationPruner
from comp_rep.utils import load_json, load_tokenizer, set_seed

set_seed(42)
SUBTASK = "copy"
CURR_FILE_PATH = Path(__file__).resolve()
TEST_DATA_PATH = CURR_FILE_PATH.parents[0] / "test_data"
PAD_LENGTH = 15


@pytest.fixture
def modelA() -> Transformer:
    return Transformer(523, 523, 2, 64, 0.2)


def test_save_and_load_cache(modelA):
    modelA.eval()
    tokenizer = load_tokenizer(TEST_DATA_PATH)

    # Cache probabilities for a small test set
    dataset_path = TEST_DATA_PATH / f"{SUBTASK}_train.csv"
    dataset_to_cache = SequenceDataset(dataset_path, tokenizer)
    cache_loader = DataLoader(
        dataset_to_cache,
        batch_size=1,
        collate_fn=CollateFunctor(max_length=PAD_LENGTH),
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
    )
    cache_save_path = TEST_DATA_PATH / f"{SUBTASK}_train.pt"
    run_caching(modelA, cache_loader, cache_save_path)
    raw_probas = torch.load(cache_save_path)

    # Load testset with the cached probabilities
    dataset_with_probabilities = SequenceDatasetWithProbabilities(
        dataset_path, tokenizer, SUBTASK
    )
    dataset_with_probabilities.cached_probabilities = raw_probas  # Need to set our local test data as the cached probas, as the Dataset loads these from a fixed path.
    loader = DataLoader(
        dataset_with_probabilities,
        batch_size=2,
        collate_fn=CollateFunctorWithProbabilities(),
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
    )

    # Run model and check that it predicts the same probabilities for the same items that were cached.
    for idx, (
        source_ids,
        source_mask,
        target_ids,
        target_mask,
        target_probabilities,
        source_str,
        target_str,
    ) in enumerate(loader):

        logits = modelA(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        ).squeeze()  # [seq_len, vocab_size]
        probas = nn.functional.softmax(logits, dim=-1)
        assert torch.all(torch.isclose(probas, target_probabilities))
