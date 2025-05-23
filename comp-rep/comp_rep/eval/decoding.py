"""
Decoding modules for sequence generation.
"""

import torch
import torch.nn as nn

from comp_rep.data_prep.dataset import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, Lang


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
        """
        Performs greedy search decoding on the given source tensor and source mask.

        Args:
            source (torch.Tensor): The input source tensor of shape [B, S], where B is the batch size and S is the sequence length.
            source_mask (torch.Tensor): The mask tensor for the source tensor of shape [B, S], where B is the batch size and S is the sequence length.

        Returns:
            Tuple[List[str], torch.Tensor]: A tuple containing a list of sentences and the target tensor.
                - sentences (List[str]): A list of sentences generated by decoding the source tensor. Each sentence is a string of tokens separated by spaces.
                - target (torch.Tensor): The target tensor of shape [B, T], where B is the batch size and T is the maximum decoding length.

        """
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
            stop |= prediction == self.pad_id
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
