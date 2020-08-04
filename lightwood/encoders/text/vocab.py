import torch
import flair
from transformers import DistilBertTokenizer

from lightwood.encoders.encoder_base import BaseEncoder


class VocabularyEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._tokenizer_class = DistilBertTokenizer
        self._pretrained_model_name = 'distilbert-base-uncased'
        self._max_len = None
        self._tokenizer = None
        self._pad_id = None

    def prepare_encoder(self, priming_data):
        self._max_len = max([len(x) for x in priming_data])
        self._tokenizer = self._tokenizer_class.from_pretrained(self._pretrained_model_name)
        self._pad_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]

    def encode(self, column_data):
        vec = []
        for text in column_data:
            encoded = self._tokenizer.encode(text[:self._max_len], add_special_tokens=True)
            encoded = torch.tensor(encoded + [self._pad_id] * (self._max_len - len(encoded)))
            vec.append(encoded)
        return torch.stack(vec)

    def decode(self, encoded_values_tensor):
        vec = []
        for encoded in encoded_values_tensor:
            decoded = self._tokenizer.decode(encoded)
            decoded = decoded.split('[PAD]')[0].rstrip().lstrip().lstrip('[CLS] ').rstrip(' [SEP]')
            vec.append(decoded)
        return vec
