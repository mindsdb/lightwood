"""
2021.03.18

## Padding changes the answer slightly in the model.

The following text encoder uses huggingface's
Distilbert. Internal benchmarks suggest
1 epoch of fine tuning is ideal [classification].
Training ONLY occurs for classification. Regression problems
are not trained, embeddings are directly generated.

See: https://huggingface.co/transformers/training.html
for further details.

Currently the model supports only distilbert.

When instantiating the DistilBertForSeq.Class object,
num_labels indicates whether you use classification or regression.

See: https://huggingface.co/transformers/model_doc/distilbert.html#distilbertforsequenceclassification
under the 'labels' command

For classification - we use num_labels = 1 + num_classes ***

If you do num_classes + 1, we reserve the LAST label
as the "unknown" label; this is different from the original
distilbert model. (prior to 2021.03)

TODOs:
+ Regression 
+ Batch encodes() tokenization step
+ Look into auto-encoding lower dimensional representations 
of the output embedding
+ Look into regression tuning (will require grad. clipping)
+ Look into tuning to the encoded space of output.
"""
import torch
from torch.utils.data import DataLoader

from lightwood.encoders.text.helpers.pretrained_helpers import TextEmbed

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log
from lightwood.helpers.torch import LightwoodAutocast

from transformers import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)


class PretrainedLang(BaseEncoder):
    """
    Pretrained language models.
    Option to train on a target encoding of choice.

    Args:
    is_target ::Bool; data column is the target of ML.
    model_name ::str; name of pre-trained model
    max_training_time ::int; seconds to train
    custom_tokenizer ::function; custom tokenizing function
    batch_size  ::int; size of batch
    max_position_embeddings ::int; max sequence length of input text
    custom_train ::Bool; If true, trains model on target procided
    frozen ::Bool; If true, freezes transformer layers during training.
    epochs ::int; number of epochs to train model with
    """

    def __init__(
        self,
        is_target=False,
        model_name="distilbert",
        custom_tokenizer=None,
        batch_size=10,
        max_position_embeddings=None,
        custom_train=True,
        frozen=False,
        epochs=1,
    ):
        super().__init__(is_target)

        self.name = model_name + " text encoder"
        log.info(self.name)

        self._max_len = max_position_embeddings
        self._custom_train = custom_train
        self._frozen = frozen
        self._batch_size = batch_size
        self._epochs = epochs

        # Model setup
        self._tokenizer = custom_tokenizer
        self._model = None
        self.model_type = None

        # TODO: Other LMs; Distilbert is a good balance of speed/performance
        self._classifier_model_class = DistilBertForSequenceClassification
        self._embeddings_model_class = DistilBertModel
        self._tokenizer_class = DistilBertTokenizerFast
        self._pretrained_model_name = "distilbert-base-uncased"

        self.device, _ = get_devices()

    def prepare(self, priming_data, training_data=None):
        """
        Prepare the encoder by training on the target.

        Training data must be a dict with "targets" avail.
        Automatically assumes this.
        """
        if self._prepared:
            raise Exception("Encoder is already prepared.")

        # TODO: Make tokenizer custom with partial function; feed custom->model
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_class.from_pretrained(
                self._pretrained_model_name
            )

        # Replaces empty strings with ''
        priming_data = [x if x is not None else "" for x in priming_data]

        # Checks training data details
        # TODO: Regression flag; currently training supported for categorical only
        output_avail = training_data is not None and len(training_data["targets"]) == 1

        if (
            self._custom_train
            and output_avail
            and (
                training_data["targets"][0]["output_type"]
                == COLUMN_DATA_TYPES.CATEGORICAL
            )
        ):
            log.info("Training model.")

            # Prepare priming data into tokenized form + attention masks
            text = self._tokenizer(priming_data, truncation=True, padding=True)

            log.info("\tOutput trained is categorical")

            if training_data["targets"][0]["encoded_output"].shape[1] > 1:
                labels = training_data["targets"][0]["encoded_output"].argmax(
                    dim=1
                )  # Nbatch x N_classes
            else:
                labels = training_data["targets"][0]["encoded_output"]

            label_size = len(set(training_data["targets"][0]["unencoded_output"])) + 1

            # Construct the model
            self._model = self._classifier_model_class.from_pretrained(
                self._pretrained_model_name,
                num_labels=label_size,
            ).to(self.device)

            # Construct the dataset for training
            xinp = TextEmbed(text, labels)
            dataset = DataLoader(xinp, batch_size=self._batch_size, shuffle=True)

            # If max length not set, adjust
            if self._max_len is None:
                self._max_len = self._model.config.max_position_embeddings

            if self._frozen:
                log.info("\tFrozen Model + Training Classifier Layers")
                """
                Freeze the base transformer model and train
                a linear layer on top
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
                num_training_steps=len(dataset) * self._epochs,
            )

            # Train model; declare optimizer earlier if desired.
            self._tune_model(
                dataset, optim=optimizer, scheduler=scheduler, n_epochs=self._epochs
            )

        else:
            log.info("Embeddings Generator only")

            self.model_type = "embeddings_generator"
            self._model = self._embeddings_model_class.from_pretrained(
                self._pretrained_model_name
            ).to(self.device)

        self._prepared = True

    def _tune_model(self, dataset, optim, scheduler, n_epochs=1):
        """
        Given a model, train for n_epochs.
        Specifically intended for tuning; it does NOT use loss/
        stopping criterion.

        model - torch.nn model;
        dataset - torch.DataLoader; dataset to train
        device - torch.device; cuda/cpu
        log - lightwood.logger.log; log.info output
        optim - transformers.optimization.AdamW; optimizer
        scheduler - scheduling params
        n_epochs - number of epochs to train

        """
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

        for epoch in range(n_epochs):
            total_loss = 0

            for batch in dataset:
                optim.zero_grad()

                with LightwoodAutocast():
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

            self._train_callback(epoch, total_loss / len(dataset))

    def _train_callback(self, epoch, loss):
        log.info(f"{self.name} at epoch {epoch+1} and loss {loss}!")

    def encode(self, column_data):
        """
        TODO: Maybe batch the text up; may take too long
        Given column data, encode the dataset.

        Currently, returns the embedding of the pre-classifier layer.

        Args:
        column_data:: [list[str]] list of text data in str form

        Returns:
        encoded_representation:: [torch.Tensor] N_sentences x Nembed_dim
        """
        if self._prepared is False:
            raise Exception("You need to first prepare the encoder.")

        # Set model to testing/eval mode.
        self._model.eval()

        encoded_representation = []
        
        with torch.no_grad():
            # Set the weights; this is GPT-2
            for text in column_data:

                # Omit NaNs
                if text is None:
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
        raise Exception("Decoder not implemented.")
