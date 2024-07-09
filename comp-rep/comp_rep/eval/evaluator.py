"""
Modules to evaluate models.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.eval.decoding import GreedySearch


def evaluate_generation(
    model: nn.Module,
    searcher: GreedySearch,
    test_loader: DataLoader,
    predictions_path: Path,
    device: str,
):
    """
    Generates predictions and evaluates them on the provided test loader.

    Args:
        model (nn.Module): The model to use for generation.
        searcher (GreedySearch): The search method for generating predictions.
        test_loader (DataLoader): The data loader for the test data.
        predictions_path (Path): The path to save the predictions.
        device (str): The device to run the model on.

    Returns:
        float: The accuracy of the predictions.
    """
    corrects = 0
    n = 0
    targets_l = []
    predictions_l = []
    outs = []
    model.to(device)
    with torch.no_grad():
        for (
            source_ids,
            source_mask,
            _,
            _,
            _,
            target_str,
        ) in tqdm(test_loader):
            source_ids = source_ids.to(device)
            source_mask = source_mask.to(device)
            sentences, _ = searcher(source_ids, source_mask)
            for t, p in zip(target_str, sentences):
                t = t.strip()
                p = p.strip()
                c = t == p
                if c:
                    corrects += 1
                n += 1
                targets_l.append(t)
                predictions_l.append(p)
                outs.append(t + "," + p + "," + str(c))
    try:
        with open(predictions_path / "predictions.csv", "w") as f:
            f.write("\n".join(outs))
    except IOError as e:
        print(f"Failed to save predictions to file.., {e}")
    finally:
        return corrects / n
