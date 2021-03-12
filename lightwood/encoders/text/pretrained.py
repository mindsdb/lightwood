"""
2021.03.11: TODO; remove print statements
Removing the sent_embedder and opting for
the CLS token only.

2021.03.10
"CLS" token instead of sent embedder in encode.

2021.03.07
TODO:
Freeze base_model in DistilBertModel
add more complicated layer and then build a model.

Pre-trained embedding model.

This deploys a hugging face transformer
and trains for a few epochs onto the target.

Once this has been completed, it provides embedding
using the updated transformer embedding.

NOTE - GPT2 does NOT have a padding token!!

Currently max_len doesn't do anything.
"""
import torch
from torch.utils.data import DataLoader

from lightwood.encoders.text.helpers.transformer_helpers import TextEmbed

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log

from transformers import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    # DistilBertConfig,
    AlbertModel,
    AlbertForSequenceClassification,
    AlbertTokenizerFast,
    # AlbertConfig,
    GPT2Model,
    GPT2ForSequenceClassification,
    GPT2TokenizerFast,
    # GPT2Config,
    BartModel,
    BartForSequenceClassification,
    BartTokenizerFast,
    # BartConfig,
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
    custom_tokenizer ::function; custom tokenizing function
    batch_size  ::int; size of batfch
    max_position_embeddings ::int; max sequence length
    custom_train ::Bool; whether to train text on target or not.
    frozen ::Bool; whether to freeze tranformer and train a linear layer head
    epochs ::int; number of epochs to train model with
    """

    def __init__(
        self,
        is_target=False,
        model_name="distilbert",
        desired_error=0.01,
        custom_tokenizer=None,
        batch_size=10,
        max_position_embeddings=None,
        custom_train=True,
        frozen=True,
        epochs=3,
    ):
        super().__init__(is_target)

        self.name = model_name + " text encoder"
        print(self.name)

        # Token/sequence treatment
        self._pad_id = None
        self._max_len = max_position_embeddings
        self._custom_train = custom_train
        self._frozen = frozen
        self._batch_size = batch_size
        self._epochs = epochs

        # Model setup
        self._tokenizer = custom_tokenizer
        self._model = None
        self.model_type = None

        if model_name == "distilbert":
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizerFast
            self._pretrained_model_name = "distilbert-base-uncased"

        elif model_name == "albert":
            self._classifier_model_class = AlbertForSequenceClassification
            self._embeddings_model_class = AlbertModel
            self._tokenizer_class = AlbertTokenizerFast
            self._pretrained_model_name = "albert-base-v2"

        elif model_name == "bart":
            self._classifier_model_class = BartForSequenceClassification
            self._embeddings_model_class = BartModel
            self._tokenizer_class = BartTokenizerFast
            self._pretrained_model_name = "facebook/bart-large"

        else:
            self._classifier_model_class = GPT2ForSequenceClassification
            self._embeddings_model_class = GPT2Model
            self._tokenizer_class = GPT2TokenizerFast
            self._pretrained_model_name = "gpt2"

        self.device, _ = get_devices()

    def prepare(self, priming_data, training_data=None):
        """
        Prepare the encoder by training on the target.
        """
        if self._prepared:
            raise Exception("Encoder is already prepared.")

        # TODO: Make tokenizer custom with partial function; feed custom->model
        if self._tokenizer is None:
            # Set the default tokenizer
            self._tokenizer = self._tokenizer_class.from_pretrained(
                self._pretrained_model_name
            )

        # Replace empty strings with ''
        priming_data = [x if x is not None else "" for x in priming_data]

        # Check style of output

        # Case 1: Categorical, 1 output
        output_type = (
            training_data is not None
            and "targets" in training_data
            and len(training_data["targets"]) == 1
            and training_data["targets"][0]["output_type"]
            == COLUMN_DATA_TYPES.CATEGORICAL
        )

        if self._custom_train and output_type:
            print("Training model.")

            # Prepare the priming data inputs with attention masks etc.
            text = self._tokenizer(priming_data, truncation=True, padding=True)

            # To train in the space, use labels as argmax.
            labels = training_data["targets"][0]["encoded_output"].argmax(
                dim=1
            )  # Nbatch x N_classes
            xinp = TextEmbed(text, labels)

            # Pad the text tokens on the left (if padding allowed)
            dataset = DataLoader(xinp, batch_size=self._batch_size, shuffle=True)

            # Construct the model
            self._model = self._classifier_model_class.from_pretrained(
                self._pretrained_model_name,
                num_labels=len(set(training_data["targets"][0]["unencoded_output"]))
                + 1,
            ).to(self.device)

            # If max length not set, adjust
            if self._max_len is None:
                if "gpt2" in self._pretrained_model_name:
                    self._max_len = self._model.config.n_positions
                else:
                    self._max_len = self._model.config.max_position_embeddings

            if self._frozen:
                print("\tFrozen Model + Training Classifier Layers")
                """
                Freeze the base transformer model and train
                a linear layer on top
                """
                # Freeze all the transformer parameters
                for param in self._model.base_model.parameters():
                    param.requires_grad = False

                optimizer_grouped_parameters = self._model.parameters()

            else:
                print("\tFine-tuning model")
                """
                Fine-tuning parameters with weight decay
                """
                # Fine-tuning weight-decay (https://huggingface.co/transformers/training.html)
                no_decay = [
                    "bias",
                    "LayerNorm.weight",
                ]  # no decay on the classifier terms.
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
                num_training_steps=len(dataset) * self._epochs,
            )

            # Train model; declare optimizer earlier if desired.
            self._train_model(
                dataset, optim=optimizer, scheduler=scheduler, n_epochs=self._epochs
            )

        else:
            print("Embeddings Generator only")

            self.model_type = "embeddings_generator"
            self._model = self._embeddings_model_class.from_pretrained(
                self._pretrained_model_name
            ).to(self.device)

        self._prepared = True

    def _train_model(self, dataset, optim=None, scheduler=None, n_epochs=4):
        """
        Given a model, train for n_epochs.

        model - torch.nn model;
        dataset - torch.DataLoader; dataset to train
        device - torch.device; cuda/cpu
        log - lightwood.logger.log; print output
        optim - transformers.optimization.AdamW; optimizer
        n_epochs - number of epochs to train

        """
        self._model.train()

        if optim is None:
            print("Setting all model params to AdamW")
            optim = AdamW(self._model.parameters(), lr=5e-5)

        for epoch in range(n_epochs):
            total_loss = 0

            for batch in dataset:
                optim.zero_grad()

                inpids = batch["input_ids"].to(self.device)
                attn = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self._model(inpids, attention_mask=attn, labels=labels)
                loss = outputs[0]

                total_loss += loss.item()

                loss.backward()
                optim.step()
                if scheduler is not None:
                    scheduler.step()

            # self._train_callback(epoch, loss.item())
            print("Epoch=", epoch + 1, "Loss=", total_loss/len(dataset))

    def _train_callback(self, epoch, loss):
        log.info(f"{self.name} at epoch {epoch+1} and loss {loss}!")

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
            for text in column_data:

                # Omit NaNs
                if text == None:
                    text = ""

                # Tokenize the text with the built-in tokenizer.
                inp = self._tokenizer.encode(
                    text, truncation=True, return_tensors="pt"
                ).to(self.device)

                output = self._model.base_model(inp).last_hidden_state[:, 0]

                # If the model has a pre-classifier layer, use this embedding.
                if hasattr(self._model, "pre_classifier"):
                    output = self._model.pre_classifier(output)

                encoded_representation.append(output.detach())

        return torch.stack(encoded_representation).squeeze(1)

    def decode(self, encoded_values_tensor, max_length=100):
        raise Exception("Decoder not implemented yet.")

    # @staticmethod
    # def _mean_norm(xinp, dim=1):
    #    """
    #    Calculates a 1 x N_embed vector by averaging all token embeddings

    #    Args:
    #    xinp ::torch.Tensor; Assumes order Nbatch x Ntokens x Nembedding
    #    dim ::int; dimension to average on
    #    """
    #    xinp = xinp[:, 1:-1, :] # Only consider word tokens and not CLS
    #    return torch.mean(xinp, dim=dim).cpu().numpy()

    # @staticmethod
    # def _cls_state(xinp):
    #    """
    #    Returns the CLS token out of the embedding.
    #    CLS is used in classification.

    #    Args:
    #        xinp ::torch.Tensor; Assumes order Nbatch x Ntokens x Nembedding
    #    """
    #    return xinp[:, 0, :].detach().cpu().numpy()
