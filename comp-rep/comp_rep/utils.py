"""
Utility functions and modules
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import Lang
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.models.model import Transformer


class ValidatePredictionPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if ".csv" in str(values):
            parser.error(
                "Only provide the path where you want to store the predictions file"
            )
        Path(values).mkdir(parents=True, exist_ok=True)
        setattr(namespace, self.dest, values)


class ValidateTaskOptions(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for v in values:
            if v not in POSSIBLE_TASKS:
                parser.error(f"{v} is not a valid task option")
        setattr(namespace, self.dest, values)


class ValidateSavePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        Path(values).mkdir(parents=True, exist_ok=True)
        if (Path.cwd() / str(values) / "model.ckpt").exists():
            logging.warning("There is already a model file saved on that path!")
        setattr(namespace, self.dest, values)


class ValidateWandbPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        Path(values).mkdir(parents=True, exist_ok=True)
        setattr(namespace, self.dest, values)


def keystoint(x: Dict) -> Dict:
    """
    Converts the keys in the input dictionary to integers if they are digits, otherwise keeps them as they are,
    and returns the modified dictionary.

    Args:
        x (Dict): The input dictionary to convert the keys.

    Returns:
        Dict: A dictionary with keys converted to integers if they are digits.
    """
    return {(int(k) if k.isdigit() else k): v for k, v in x.items()}


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in torch, numpy, and random libraries.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_tokenizer_dict(input_lang: Lang, output_lang: Lang) -> Dict:
    """
    Serializes the tokenizers

    Args:
        input_lang (Lang): The input language representation.
        output_lang (Lang): The output language representation.

    Returns:
        Dict: The language mapping.
    """
    return {
        "input_language": {
            "index2word": input_lang.index2word,
            "word2index": input_lang.word2index,
        },
        "output_language": {
            "index2word": output_lang.index2word,
            "word2index": output_lang.word2index,
        },
    }


def save_tokenizer(path: Path, tokenizer: dict) -> None:
    """
    Saves the index2word and word2index dicts from the languages

    Args:
        path (Path): The path to the tokenizer file.
        tokenizer (dict): The dictionary representing the tokenizer.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path / "tokenizers.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False, indent=4)


def load_tokenizer(path: Path) -> Dict:
    """
    Loads the index2word and word2index dicts for both languages

    Args:
        path (Path): The path to the tokenizer file.

    Returns:
        Dict: The tokenizer.
    """
    with open(path / "tokenizers.json", "r") as f:
        tokenizers = json.load(f, object_hook=keystoint)
    return tokenizers


def setup_logging(verbosity: int = 1) -> None:
    """
    Set up logging based on the verbosity level.

    Args:
        verbosity (int): Verbosity level.
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    else:
        raise ValueError(
            f"Invalid log-level specified: {verbosity}! Should be 0, 1, or 2."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def create_transformer_from_checkpoint(model_path: Path) -> nn.Module:
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    input_vocabulary_size = vars(checkpoint["hyper_parameters"]["args"])[
        "input_vocabulary_size"
    ]
    output_vocabulary_size = vars(checkpoint["hyper_parameters"]["args"])[
        "output_vocabulary_size"
    ]
    num_transformer_layers = vars(checkpoint["hyper_parameters"]["args"])["layers"]
    hidden_size = vars(checkpoint["hyper_parameters"]["args"])["hidden_size"]
    dropout = vars(checkpoint["hyper_parameters"]["args"])["dropout"]
    base_model = Transformer(
        input_vocabulary_size,
        output_vocabulary_size,
        num_transformer_layers,
        hidden_size,
        dropout,
    )
    return base_model


def load_model(
    path: Path,
    is_masked: bool,
    model: Optional[nn.Module],
):
    if is_masked:
        pl_pruner = LitPrunedModel.load_from_checkpoint(path, model=model)
        model = pl_pruner.model
        pl_pruner.pruner.activate_ticket()
        pl_pruner.pruner.compute_and_update_masks()
    else:
        pl_transformer = LitTransformer.load_from_checkpoint(path)
        model = pl_transformer.model
    return model


def load_transformed_model_no_update(
    path: Path,
    model: Optional[nn.Module],
):
    pl_pruner = LitPrunedModel.load_from_checkpoint(path, model=model)
    model = pl_pruner.model
    return model


def save_list_to_csv(file_path: Path, data: List[str]) -> None:
    """
    Save a list of strings to a CSV file.

    Parameters:
    file_path (str): Name of the CSV file to save.
    data (List[str]): List of strings to save.
    """
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="") as csvfile:
        for item in data:
            csvfile.write(item + "\n")
