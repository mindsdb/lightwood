"""
2021.02.16

The current implementation will try out different text models.

Similar to DistilBERT; DOES not train by default.
Using pre-trained models from huggingface.

NOTEs:
-- Albert needs to be padded
-- using fast tokenizers;
-- T5 is a good for particular targets possibly

Fine tuning: https://github.com/huggingface/transformers/issues/4749

I'm probably going to need a superlongform text cutter...
Line 216 is temporary hack.
"""

from functools import partial

import torch
from torch.utils.data import DataLoader

from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, ENCODER_AIM
from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.shapes import *
from lightwood.api.gym import Gym
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log

from transformers import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    #DistilBertConfig,
    AlbertModel,
    AlbertForSequenceClassification,
    AlbertTokenizerFast,
    #AlbertConfig,
    GPT2Model,
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    #GPT2Config,
    BartModel,
    BartForSequenceClassification,
    BartTokenizerFast,
    #BartConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)


class PretrainedLang(BaseEncoder):
    """
    Pretrained language models.
    Option to train on a target encoding of choice.

    The "sent_embedder" parameter refers to a function to make
    sentence embeddings, given a 1 x N_tokens x N_embed input

    Args:
    is_target ::Bool; data column is the target of ML.
    model_name ::str; name of pre-trained model
    desired_error ::float
    max_training_time ::int; seconds to train
    custom_train ::Bool; whether to train text on target or not.
    custom_tokenizer ::function; custom tokenizing function
    sent_embedder ::str; make a sentence embedding from seq of word embeddings
                         default, sum all tokens and average
    max_position_embeddings ::int; max sequence length
    """

    def __init__(
        self,
        is_target=False,
        model_name="gpt2",
        desired_error=0.01,
        max_training_time=7200,
        custom_train=False,
        custom_tokenizer=None,
        sent_embedder='mean_norm',
        #max_position_embeddings=512,
    ):
        super().__init__(is_target)

        self.name = model_name + " text encoder"

        # Token/sequence treatment
        self._pad_id = None
        #self._max_len = max_position_embeddings
        self._max_ele = None
        self._custom_train = custom_train

        # Model details
        self.desired_error = desired_error
        self.max_training_time = max_training_time
        self._head = None

        # Model setup
        self._tokenizer = custom_tokenizer
        self._model = None
        self._model_type = None

        if model_name == "distilbert":
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizerFast
            self._pretrained_model_name = "distilbert-base-uncased"
            #self._enc_config = DistilBertConfig(max_position_embedding=self._max_len )
        elif model_name == "albert":
            self._classifier_model_class = AlbertForSequenceClassification
            self._embeddings_model_class = AlbertModel
            self._tokenizer_class = AlbertTokenizerFast
            self._pretrained_model_name = "albert-base-v2"
            #self._enc_config = AlbertConfig(max_position_embeddings=self._max_len )
        elif model_name == "bart":
            self._classifier_model_class = BartForSequenceClassification
            self._embeddings_model_class = BartModel
            self._tokenizer_class = BartTokenizerFast
            self._pretrained_model_name = "facebook/bart-large"
            #self._enc_config = BartConfig(max_position_embeddings=self._max_len )
        else:
            self._classifier_model_class = GPT2ForSequenceClassification
            self._embeddings_model_class = GPT2Model
            self._tokenizer_class = GPT2TokenizerFast
            self._pretrained_model_name = "gpt2"
            #self._enc_config = GPT2Config(n_positions=self._max_len)

        # Type of sentence embedding
        if sent_embedder == 'last_token':
            self._sent_embedder = self._mean_norm
        else:
            self._sent_embedder = self._mean_norm

        self.device, _ = get_devices()

    def to(self, device, available_devices):
        """ Set torch device to CPU/CUDA """
        self._model = self._model.to(self.device)

        if self._head is not None:
            self._head = self._head.to(self.device)

        return self

    def prepare(self):
        """
        Prepare the text encoder to convert text -> feature vector

        Args:
        custom_tok ::Bool; whether to tokenize prior to passing to language model tokenizer.

        """
        if self._prepared:
            raise Exception("Encoder is already prepared.")

        if self._tokenizer is None:
            # Set the tokenizer
            self._tokenizer = self._tokenizer_class.from_pretrained(
                self._pretrained_model_name
            )
        # TODO: add else; create a partial function to map LANG_TOK(CUSTOM_TOK(x))

        if self._custom_train is True:
            # TODO: Custom train on the target
            raise NotImplementedError("TODO; train fxn not implemented")
        else:
            self._model_type = "embeddings_generator"
            self._model = self._embeddings_model_class.from_pretrained(
                self._pretrained_model_name
            ).to(self.device)
            if "gpt2" in self._pretrained_model_name:
                self._max_len = self._model.config.n_positions
            else:
                self._max_len = self._model.config.max_position_embeddings

        # Depending on model type, get forward pass

        self._prepared = True

    def encode(self, column_data):
        """
        TODO: Maybe batch the text up; may take too long
        Given column data, encode the dataset
        Tokenizer should have a length cap!!

        Args:
        column_data:: [list[str]] list of text data in str form

        Returns:
        encoded_representation:: [torch.Tensor] N_sentences x Nembed_dim
        """
        encoded_representation = []

        # Freeze training mode while encoding
        self._model.eval()

        with torch.no_grad():
            # Set the weights; this is GPT-2
            if self._model_type == "embeddings_generator":
                for text in column_data:

                    # Omit NaNs
                    if text == None:
                        text = ''

                    # Tokenize the text with the built-in tokenizer.
                    inp = self._tokenizer.encode(text, return_tensors="pt").to(self.device)

                    # TODO - try different accumulation techniques?
                    # TODO: Current hack is to keep the first max len
                    inp = inp[:, :self._max_len]

                    output = self._model(inp).last_hidden_state
                    output = self._sent_embedder(output.to(self.device))

                    encoded_representation.append(output)

        return torch.Tensor(encoded_representation).squeeze(1)

    def decode(self, encoded_values_tensor, max_length=100):
        raise Exception("Decoder not implemented yet.")

    def _train_callback(self, error, real_buff, predicted_buff):
        log.info(f"{self.name} reached a loss of {error} while training !")

    @staticmethod
    def _mean_norm(xinp, dim=1):
        """
        Calculates a 1 x N_embed vector by averaging all token embeddings

        Args:
        xinp ::torch.Tensor; Assumes order Nbatch x Ntokens x Nembedding
        dim ::int; dimension to average on
        """
        return torch.mean(xinp, dim=dim).cpu().numpy()

    @staticmethod
    def _last_state(xinp):
        """
        Returns the last token in the sentence only

        Args:
            xinp ::torch.Tensor; Assumes order Nbatch x Ntokens x Nembedding
        """
        return xinp[:, -1, :].cpu().numpy()


