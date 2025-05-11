import os
import time
from typing import Iterable
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader
import pandas as pd
from type_infer.dtype import dtype
from transformers import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

from lightwood.encoder.text.helpers.pretrained_helpers import TextEmbed
from lightwood.helpers.device import get_device_from_name
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.general import is_none


class PretrainedLangEncoder(BaseEncoder):
    is_trainable_encoder: bool = True

    """
    Creates a contextualized embedding to represent input text via the [CLS] token vector from DistilBERT (transformers). (Sanh et al. 2019 - https://arxiv.org/abs/1910.01108).

    In certain text tasks, this model can use a transformer to automatically fine-tune on a class of interest (providing there is a 2 column dataset, where the input column is text).

    """ # noqa

    def __init__(
        self,
        stop_after: float,
        is_target: bool = False,
        batch_size: int = 10,
        max_position_embeddings: int = None,
        frozen: bool = False,
        epochs: int = 1,
        output_type: str = None,
        embed_mode: bool = True,
        device: str = '',
    ):
        """
        :param is_target: Whether this encoder represents the target. NOT functional for text generation yet.
        :param batch_size: size of batch while fine-tuning
        :param max_position_embeddings: max sequence length of input text
        :param frozen: If True, freezes transformer layers during training.
        :param epochs: number of epochs to train model with
        :param output_type: Data dtype of the target; if categorical/binary, the option to return logits is possible.
        :param embed_mode: If True, assumes the output of the encode() step is the CLS embedding (this can be trained or not). If False, returns the logits of the tuned task.
        :param device: name of the device that get_device_from_name will attempt to use.
        """ # noqa
        super().__init__(is_target)

        self.output_type = output_type
        self.name = "distilbert text encoder"

        self._max_len = max_position_embeddings
        self._frozen = frozen
        self._batch_size = batch_size
        self._epochs = epochs
        self._patience = 3  # measured in batches rather than epochs
        self._val_loss_every = 5  # how many batches to wait before checking val loss. If -1, will check train loss instead of val for early stopping.  # noqa
        self._tr_loss_every = 2  # same as above, but only applies if `_val_loss_every` is set to -1

        # Model setup
        self._model = None
        self.model_type = None

        self._classifier_model_class = DistilBertForSequenceClassification
        self._embeddings_model_class = DistilBertModel
        self._pretrained_model_name = "distilbert-base-uncased"
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(self._pretrained_model_name)

        self.device = get_device_from_name(device)

        self.stop_after = stop_after

        self.embed_mode = embed_mode
        self.uses_target = True
        self.output_size = None

        if self.embed_mode:
            log.info("Embedding mode on. [CLS] embedding dim output of encode()")
        else:
            log.info("Embedding mode off. Logits are output of encode()")

    def prepare(
        self,
        train_priming_data: pd.Series,
        dev_priming_data: pd.Series,
        encoded_target_values: torch.Tensor,
    ):
        """
        Fine-tunes a transformer on the priming data.

        Transformer is fine-tuned with weight-decay on training split. 
        
        Train + Dev are concatenated together and a transformer is then fine tuned with weight-decay applied on the transformer parameters. The option to freeze the underlying transformer and only train a linear layer exists if `frozen=True`. This trains faster, with the exception that the performance is often lower than fine-tuning on internal benchmarks.
        
        :param train_priming_data: Text data in the train set
        :param dev_priming_data: Text data in the dev set
        :param encoded_target_values: Encoded target labels in Nrows x N_output_dimension
        """ # noqa
        if self.is_prepared:
            raise Exception("Encoder is already prepared.")

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'

        # remove empty strings (`None`s for dtype `object`)
        filtered_tr = train_priming_data[~train_priming_data.isna()]
        filtered_dev = dev_priming_data[~dev_priming_data.isna()]

        if filtered_dev.shape[0] > 0:
            priming_data = pd.concat([filtered_tr, filtered_dev]).tolist()
            val_size = (len(dev_priming_data)) / len(train_priming_data)
        else:
            priming_data = filtered_tr.tolist()
            val_size = 0.1  # leave out 0.1 for validation

        # Label encode the OHE/binary output for classification
        labels = encoded_target_values.argmax(dim=1)

        # Split into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(priming_data, labels, test_size=val_size)

        # If classification, then fine-tune
        if self.output_type in (dtype.categorical, dtype.binary):
            log.info("Training model.\n\tOutput trained is categorical")

            # Prepare priming data into tokenized form + attention masks
            training_text = self._tokenizer(train_texts, truncation=True, padding=True)
            validation_text = self._tokenizer(val_texts, truncation=True, padding=True)

            # Construct the model
            self._model = self._classifier_model_class.from_pretrained(
                self._pretrained_model_name,
                num_labels=len(encoded_target_values[0]),  # max classes to test
            ).to(self.device)

            # Construct the dataset for training
            xinp = TextEmbed(training_text, train_labels)
            train_dataset = DataLoader(xinp, batch_size=self._batch_size, shuffle=True)

            # Construct the dataset for validation
            xvalinp = TextEmbed(validation_text, val_labels)
            val_dataset = DataLoader(xvalinp, batch_size=self._batch_size, shuffle=True)

            # Set max length of input string; affects input to the model
            if self._max_len is None:
                self._max_len = self._model.config.max_position_embeddings

            if self._frozen:
                log.info("\tFrozen Model + Training Classifier Layers")
                """
                Freeze the base transformer model and train a linear layer on top
                """
                # Freeze all the transformer parameters
                for param in self._model.base_model.parameters():
                    param.requires_grad = False

                optimizer_grouped_parameters = self._model.parameters()

            else:
                log.info("\tFine-tuning model")
                """
                Fine-tuning parameters with weight decay
                """
                no_decay = [
                    "bias",
                    "LayerNorm.weight",
                ]  # decay on all terms EXCLUDING bias/layernorms
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in self._model.named_parameters()
                            if not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.01,
                    },
                    {
                        "params": [
                            p
                            for n, p in self._model.named_parameters()
                            if any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,  # default value for GLUE
                num_training_steps=len(train_dataset) * self._epochs,
            )

            # Train model; declare optimizer earlier if desired.
            self._tune_model(
                train_dataset, val_dataset, optim=optimizer, scheduler=scheduler, n_epochs=self._epochs
            )

        else:
            log.info("Target is not classification; Embeddings Generator only")

            self.model_type = "embeddings_generator"
            self._model = self._embeddings_model_class.from_pretrained(
                self._pretrained_model_name
            ).to(self.device)

            # TODO: Not a great flag
            # Currently, if the task is not classification, you must have an embedding generator only
            if self.embed_mode is False:
                log.info("Embedding mode must be ON for non-classification targets.")
                self.embed_mode = True

        self.is_prepared = True
        encoded = self.encode(priming_data[0:1])
        self.output_size = len(encoded[0])

    def _tune_model(self, train_dataset, val_dataset, optim, scheduler, n_epochs=1):
        """
        Given a model, tune for n_epochs.

        train_dataset - torch.DataLoader; dataset to train
        val_dataset - torch.DataLoader; dataset used to compute validation loss + early stopping
        optim - transformers.optimization.AdamW; optimizer
        scheduler - scheduling params
        n_epochs - max number of epochs to train for, provided there is no early stopping

        """ # noqa
        self._model.train()

        if optim is None:
            log.info("No opt. provided, setting all params with AdamW.")
            optim = AdamW(self._model.parameters(), lr=5e-5)
        else:
            log.info("Optimizer provided")

        if scheduler is None:
            log.info("No scheduler provided.")
        else:
            log.info("Scheduler provided.")

        best_tr_loss = best_val_loss = float("inf")
        tr_loss_queue = deque(maxlen=self._patience)
        patience_counter = self._patience

        started = time.time()
        for epoch in range(n_epochs):
            total_loss = 0

            for bidx, batch in enumerate(train_dataset):
                optim.zero_grad()

                with LightwoodAutocast():
                    loss = self._call(batch)

                tr_loss_queue.append(loss.item())
                total_loss += loss.item()

                loss.backward()
                optim.step()
                if scheduler is not None:
                    scheduler.step()
                if time.time() - started > self.stop_after:
                    break

                # val-based early stopping
                if False and (self._val_loss_every != -1) and (bidx % self._val_loss_every == 0):
                    self._model.eval()
                    val_loss = 0

                    for vbatch in val_dataset:
                        val_loss += self._call(vbatch).item()

                    log.info(
                        f"Epoch {epoch + 1} train batch {bidx + 1} - "
                        f"Validation loss: {val_loss / len(val_dataset)}"
                    )
                    if val_loss / len(val_dataset) >= best_val_loss:
                        break

                    best_val_loss = val_loss / len(val_dataset)
                    self._model.train()

                # train-based early stopping
                elif False and (bidx + 1) % self._tr_loss_every == 0:
                    self._model.eval()

                    tr_loss = np.average(tr_loss_queue)
                    log.info(f"Epoch {epoch} train batch {bidx} - Train loss: {tr_loss}")  # noqa
                    self._model.train()

                    if tr_loss >= best_tr_loss and patience_counter == 0:
                        break
                    elif patience_counter > 0:
                        patience_counter -= 1
                    elif tr_loss < best_tr_loss:
                        best_tr_loss = tr_loss
                        patience_counter = self._patience

            if time.time() - started > self.stop_after:
                break
            self._train_callback(epoch, total_loss / len(train_dataset))

    def _call(self, batch):
        inpids = batch["input_ids"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        outputs = self._model(inpids, attention_mask=attn, labels=labels)
        loss = outputs[0]
        return loss

    def _train_callback(self, epoch, loss):
        log.info(f"{self.name} at epoch {epoch + 1} and loss {loss}!")

    def encode(self, column_data: Iterable[str]) -> torch.Tensor:
        """
        Converts each text example in a column into encoded state. This can be either a vector embedding of the [CLS] token (represents the full text input) OR the logits prediction of the output.

        The transformer model is of form:
        transformer base + pre-classifier linear layer + classifier layer

        The embedding returned is of the [CLS] token after the pre-classifier layer; from internal testing, we found the latent space most highly separated across classes. 

        If the encoder represents the logits in classification, returns a soft-maxed output of the class vector.

        :param column_data: List of text data as strings
        :returns: Embedded vector N_rows x Nembed_dim OR logits vector N_rows x N_classes depending on if `embed_mode` is True or not.
        """ # noqa
        if self.is_prepared is False:
            raise Exception("You need to first prepare the encoder.")

        # Set model to testing/eval mode.
        self._model.eval()

        encoded_representation = []

        with torch.no_grad():
            # Set the weights; this is GPT-2
            for text in column_data:

                # Omit NaNs
                if is_none(text):
                    text = ""

                # Tokenize the text with the built-in tokenizer.
                inp = self._tokenizer.encode(
                    text, truncation=True, return_tensors="pt"
                ).to(self.device)

                if self.embed_mode:  # Embedding mode ON; return [CLS]
                    output = self._model.base_model(inp).last_hidden_state[:, 0]

                    # If the model has a pre-classifier layer, use this embedding.
                    if hasattr(self._model, "pre_classifier"):
                        output = self._model.pre_classifier(output)

                else:  # Embedding mode off; return classes
                    output = self._model(inp).logits

                encoded_representation.append(output.detach())

        return torch.stack(encoded_representation).squeeze(1).to('cpu')

    def decode(self, encoded_values_tensor, max_length=100):
        """
        Text generation via decoding is not supported.
        """ # noqa
        raise Exception("Decoder not implemented.")

    def to(self, device, available_devices):
        """
        Converts encoder models to device specified (CPU/GPU)

        Transformers are LARGE models, please run on GPU for fastest implementation.
        """ # noqa
        for v in vars(self):
            attr = getattr(self, v)
            if isinstance(attr, torch.nn.Module):
                attr.to(device)
        return self
