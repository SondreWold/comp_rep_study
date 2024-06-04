from collections import defaultdict
from torch.utils.data import Dataset
from typing import Optional
import pathlib
import torch
import torch.nn.functional as F

Pairs = list[list[str, str]]
DatasetItem = tuple[torch.Tensor, torch.Tensor, str, str]
CollatedItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
# The few samples that are longer than this are skipped in order to save resources on the sequences length
MAX_SAMPLE_LENGTH = 256


class Lang:
    def __init__(self, name: str,
                 word2index: Optional[dict] = None,
                 index2word: Optional[dict] = None) -> None:
        if index2word is None:
            index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        if word2index is None:
            word2index = {}
        self.name = name
        self.word2count = defaultdict(int)
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = 3  # Count PAD, SOS and EOS

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class SequenceDataset(Dataset):
    def __init__(self, path: pathlib.Path, tokenizer: Optional[dict] = None) -> None:
        self.pairs = self.read_pairs(path)

        self.input_language = Lang("INPUT")
        self.output_language = Lang("OUTPUT")
        for ip, op in self.pairs:
            self.input_language.add_sentence(ip)
            self.output_language.add_sentence(op)

    def read_pairs(self, path: pathlib.Path) -> Pairs:
        try:
            lines = open(path, encoding="utf-8").read().strip().split("\n")
        except IOError as e:
            print("Failed to open data file: ", e)

        pairs = []
        for line in lines:
            inl, outl = line.split(';')
            inl = inl.strip()
            outl = outl.strip()
            pairs.append([inl, outl])
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def indexesFromSentence(self, lang: Lang, sentence: str) -> list[int]:
        return [lang.word2index[word] for word in sentence.split(' ')]

    def __getitem__(self, idx) -> DatasetItem:
        x, y = self.pairs[idx]
        input_tensor = self.indexesFromSentence(self.input_language, x) + [EOS_TOKEN]
        output_tensor = [SOS_TOKEN] + self.indexesFromSentence(self.output_language, y) + [EOS_TOKEN]
        return torch.tensor(input_tensor), torch.tensor(output_tensor), x, y


class CollateFunctor:
    def __init__(self) -> None:
        self.pad_id = PAD_TOKEN

    def __call__(self, sentences: list) -> CollatedItem:
        source_ids, target_ids, source_str, target_str = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(source_ids)
        target_ids, target_mask = self.collate_sentences(target_ids)
        return source_ids, source_mask, target_ids, target_mask, source_str, target_str

    def collate_sentences(self, sentences: list) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [sentence.size(0) for sentence in sentences]
        max_length = max(lengths)
        if max_length == 1:  # Some samples are only one token
            max_length += 1
        subword_ids = torch.stack([
            F.pad(sentence, (0, max_length - length), value=self.pad_id)
            for length, sentence in zip(lengths, sentences)
        ])
        attention_mask = subword_ids == self.pad_id
        return subword_ids, attention_mask


if __name__ == "__main__":
    test_path = "../data/base_tasks/pcfgs_train.csv"
    data = SequenceDataset(path=test_path)
    print(data[0])
