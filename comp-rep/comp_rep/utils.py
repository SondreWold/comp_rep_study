from typing import Dict

from dataset import Lang


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
