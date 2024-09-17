"""
Modules to evaluate models.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.metrics import jsd_faithfulness


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    searcher: GreedySearch,
    test_loader: DataLoader,
    predictions_path: Optional[Path] = None,
    device: Optional[str] = "cuda:0",
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
    for batch in tqdm(test_loader):
        source_ids = batch[0].to(
            device
        )  # With indexing we do not have to do a type check on the dataset type
        source_mask = batch[1].to(device)
        target_str = batch[-1]
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

    if predictions_path is not None:
        try:
            Path(predictions_path).mkdir(parents=True, exist_ok=True)
            with open(predictions_path / "predictions.csv", "w") as f:
                f.write("\n".join(outs))
        except IOError as e:
            print(f"Failed to save predictions to file.., {e}")
        finally:
            return corrects / n

    return corrects / n


@torch.no_grad()
def evaluate_task_faithfulness(
    model: nn.Module,
    test_loader: DataLoader,
    device: Union[str, torch.device] = "cuda:0",
) -> float:
    """
    Evaluates the faithfulness of a given model on a test dataset.

    Args:
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): The test dataset loader.
        device (Union[str, torch.device], optional): The device to be used for evaluation. Defaults to "cuda:0".

    Returns:
        float: The average faithfulness score of the model over the test set.
    """
    model = model.to(device)
    model.eval()

    total_faithfulness_score: float = 0.0
    n_batches: int = 0

    for batch in tqdm(test_loader):
        source_ids, source_mask, target_ids, target_mask, target_probabilities, _, _ = (
            batch
        )

        # Move tensors to device
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        target_ids = target_ids.to(device)
        target_mask = target_mask.to(device)
        target_probabilities = target_probabilities.to(device)

        # Left shift the targets so that the last token predicts the EOS
        model_logits = model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]

        faithfulness_score = jsd_faithfulness(
            p_logits=model_logits, q_probs=target_probabilities, eps=1e-10
        )
        total_faithfulness_score += faithfulness_score
        n_batches += 1

    return total_faithfulness_score / n_batches


def eval_task(
    task_name: str,
    model: nn.Module,
    tokenizer: Dict,
    device: str,
    eval_data_path: Path,
    output_dir: Path,
) -> float:
    """
    Evaluates the performance of a given model on a specific task.

    Args:
        task_name (str): The name of the task to be evaluated.
        model (nn.Module): The model to be evaluated.
        tokenizer (Dict): The tokenizer used for the task.
        device (str): The device to be used for evaluation.
        eval_data_path (Path): The path to the evaluation data.
        output_dir (Path): The directory where the evaluation results will be saved.

    Returns:
        float: The accuracy of the model on the given task.
    """
    logging.info(f"Evaluating function: {task_name}")

    eval_dataset = SequenceDataset(eval_data_path, tokenizer=tokenizer)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=64,
        collate_fn=CollateFunctor(),
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )
    searcher = GreedySearch(model, eval_dataset.output_language)

    accuracy = evaluate_generation(model, searcher, eval_loader, output_dir, device)

    return accuracy
