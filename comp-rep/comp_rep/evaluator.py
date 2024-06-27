import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dataset import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, Lang, SequenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluation script")
    parser.add_argument("--eval_data_path", type=Path)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


class GreedySearch:
    def __init__(self, model: nn.Module, output_language: Lang, max_length: int = 512):
        self.model = model
        self.model.eval()
        self.max_length = max_length
        self.output_language = output_language

        self.sos_id = SOS_TOKEN
        self.eos_id = EOS_TOKEN
        self.pad_id = PAD_TOKEN

    @torch.no_grad()
    def __call__(self, source: torch.Tensor, source_mask: torch.Tensor):
        source_encoding = self.model.encode_source(source, source_mask)
        # [B, 1], Start tokens to initiate generation
        target = torch.full(
            [source_encoding.size(0), 1], fill_value=self.sos_id, device=source.device
        )
        # [B], indicators for where to stop generating / EOS reached
        stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)

        for _ in range(self.max_length):
            prediction = self.model.decode_step(
                target, None, source_encoding, source_mask
            )[:, -1]
            # Add pad tokens to the positions after EOS token has been reached in a previous step
            prediction = torch.where(stop, self.pad_id, prediction.argmax(-1))
            # Set stop to True where the prediction is equal to EOS token
            stop |= prediction == self.eos_id
            target = torch.cat([target, prediction.unsqueeze(1)], dim=1)
            if stop.all():
                break

        sentences = []
        for batch in target:
            sent = ""
            for token in batch[1:]:
                if token.item() == self.eos_id:
                    break
                sent += self.output_language.index2word[token.item()] + " "
            sentences.append(sent)
        return sentences, target


def evaluate_generation(
    model: nn.Module,
    searcher: GreedySearch,
    test_loader: DataLoader,
    predictions_path: Path,
):
    """
    Generates the predictions on the provided test loader.
    """
    corrects = 0
    n = 0
    targets_l = []
    predictions_l = []
    outs = []
    with torch.no_grad():
        for (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            source_str,
            target_str,
        ) in tqdm(test_loader):
            source_ids = source_ids.to(DEVICE)
            source_mask = source_mask.to(DEVICE)
            sentences, predictions = searcher(source_ids, source_mask)
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


def load_eval_data(path: Path, tokenizer: dict) -> SequenceDataset:
    dataset = SequenceDataset(path, tokenizer)
    return dataset


def main(args: argparse.Namespace) -> None:
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning evaluation with the config: \n{config_string}")
    # TODO: When run as a stand alone, this requires first implementing saving/loading


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main(parse_args())
