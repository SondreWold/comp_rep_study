import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from dataset import Lang


def keystoint(x):
    return {(int(k) if k.isdigit() else k): v for k, v in x.items()}


class ValidatePredictionPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if ".csv" in str(values):
            parser.error(
                "Only provide the path where you want to store the predictions file"
            )
        Path(values).mkdir(parents=True, exist_ok=True)
        setattr(namespace, self.dest, values)


class ValidateSavePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        Path(values).mkdir(parents=True, exist_ok=True)
        if (Path.cwd() / str(values) / "model.ckpt").exists():
            print("WARNING: There is already a model file saved on that path!")
        setattr(namespace, self.dest, values)


def validate_args(args: argparse.Namespace):
    if args.eval and not args.predictions_path:
        print(
            "For eval mode, an output path for the predictions must be set with --predictions_path"
        )
        exit(0)


def set_seed(seed: int) -> None:
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
    """
    with open(path / "tokenizers.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False, indent=4)


def load_tokenizer(path: Path) -> Dict:
    """
    Loads the index2word and word2index dicts for both languages
    """
    with open(path / "tokenizers.json", "r") as f:
        tokenizers = json.load(f, object_hook=keystoint)
    return tokenizers
