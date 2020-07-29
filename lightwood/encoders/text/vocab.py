import flair
from transformers import DistilBertTokenizer

from lightwood.encoders.encoder_base import BaseEncoder


class VocabularyEncoder(BaseEncoder):
    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._tokenizer_class = DistilBertTokenizer
        self._pretrained_model_name = 'distilbert-base-uncased'
        self._max_len = None

    def prepare_encoder(self, priming_data):
        print(priming_data)
        self._max_len = max([len(x) for x in priming_data])
        self._tokenizer = self._tokenizer_class.from_pretrained(self._pretrained_model_name)
        self._pad_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]

    def encode(self, column_data):
        encoded = self._tokenizer.encode(text[:self._max_len], add_special_tokens=True)
        print(encoded)
        return encoded

    def decode(self, encoded_values_tensor):
        decode = self._tokenizer.decode(encoded_values_tensor, add_special_tokens=True)
        return decode
