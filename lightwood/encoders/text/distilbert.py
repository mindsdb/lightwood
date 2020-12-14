from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertModel,
    DistilBertForSequenceClassification,
    AlbertModel,
    AlbertForSequenceClassification,
    DistilBertTokenizer,
    AlbertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, ENCODER_AIM
from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.shapes import *
from lightwood.api.gym import Gym
from lightwood.helpers.torch import LightwoodAutocast
from lightwood.helpers.device import get_devices
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log


class DistilBertEncoder(BaseEncoder):
    def __init__(self, is_target=False, aim=ENCODER_AIM.BALANCE):
        super().__init__(is_target)
        self.name = 'Text Transformer Encoder'
        self._tokenizer = None
        self._model = None
        self._pad_id = None
        self._max_len = None
        self._max_ele = None
        self._model_type = None
        self.desired_error = 0.01
        self.max_training_time = 7200
        self._head = None
        # Possible: speed, balance, accuracy
        self.aim = aim

        if self.aim == ENCODER_AIM.SPEED:
            # uses more memory, takes very long to train and outputs weird debugging statements to the command line,
            # consider waiting until it gets better or try to investigate why this happens
            # (changing the pretrained model doesn't seem to help)
            self._classifier_model_class = AlbertForSequenceClassification
            self._embeddings_model_class = AlbertModel
            self._tokenizer_class = AlbertTokenizer
            self._pretrained_model_name = 'albert-base-v2'
        if self.aim == ENCODER_AIM.BALANCE:
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizer
            self._pretrained_model_name = 'distilbert-base-uncased'
        if self.aim == ENCODER_AIM.ACCURACY:
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizer
            self._pretrained_model_name = 'distilbert-base-uncased'

        self.device, _ = get_devices()

    def _train_callback(self, error, real_buff, predicted_buff):
        log.info(f'{self.name} reached a loss of {error} while training !')

    @staticmethod
    def categorical_train_function(model, data, gym, test=False):
        input, real = data
        input = input.to(gym.device)
        labels = torch.tensor([torch.argmax(x) for x in real]).to(gym.device)

        with LightwoodAutocast():
            outputs = gym.model(input, labels=labels)
            loss, logits = outputs[:2]

        if not test:
            loss.backward()
            gym.optimizer.step()
            gym.scheduler.step()
            gym.optimizer.zero_grad()
        return loss

    @staticmethod
    def numerical_train_function(model, data, gym, backbone, test=False):
        input, real = data

        backbone = backbone.eval()

        with torch.no_grad():
            input = input.to(gym.device)
            real = real.to(gym.device)

            embeddings = backbone(input)[0][:, 0, :]

        with LightwoodAutocast():
            outputs = gym.model(embeddings)
            loss = gym.loss_criterion(outputs, real)

        if not test:
            loss.backward()
            gym.optimizer.step()
            gym.scheduler.step()
            gym.optimizer.zero_grad()

        return loss

    def to(self, device, available_devices):
        self._model = self._model.to(self.device)

        if self._head is not None:
            self._head = self._head.to(self.device)

        return self

    def prepare(self, priming_data, training_data=None):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        priming_data = [x if x is not None else '' for x in priming_data]

        self._tokenizer = self._tokenizer_class.from_pretrained(self._pretrained_model_name)
        self._pad_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]
        # @TODO: Support multiple targets if they are all categorical
        # or train for the categorical target if it's a mix (maybe ?)

        # @TODO: Attach a language modeling head and/or use GPT2
        # and/or provide outputs better suited to a LM head (which will be the mixer) if the output if text
        if training_data is not None and 'targets' in training_data and len(training_data['targets']) == 1 and training_data['targets'][0]['output_type'] == COLUMN_DATA_TYPES.CATEGORICAL:

            self._model_type = 'classifier'
            self._model = self._classifier_model_class.from_pretrained(self._pretrained_model_name, num_labels=len(
                set(training_data['targets'][0]['unencoded_output'])) + 1).to(self.device)
            self._max_len = min(max([len(x) for x in priming_data]), self._model.config.max_position_embeddings)
            batch_size = 10

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self._model.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.000001},
                {'params': [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=10, num_training_steps=len(priming_data) * 15 / 20)

            gym = Gym(model=self._model, optimizer=optimizer, scheduler=scheduler,
                      loss_criterion=None, device=self.device, name=self.name)

            input = [self._tokenizer.encode(x[:self._max_len], add_special_tokens=True) for x in priming_data]
            tokenized_max_len = max([len(x) for x in input])
            input = torch.tensor([x + [self._pad_id] * (tokenized_max_len - len(x)) for x in input])

            real = training_data['targets'][0]['encoded_output']

            merged_data = list(zip(input, real))

            train_data_loader = DataLoader(
                merged_data[:int(len(merged_data) * 9 / 10)], batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(
                merged_data[int(len(merged_data) * 9 / 10):], batch_size=batch_size, shuffle=True)

            best_model, error, training_time = gym.fit(train_data_loader=train_data_loader,
                                                       test_data_loader=test_data_loader,
                                                       desired_error=self.desired_error,
                                                       max_time=self.max_training_time,
                                                       callback=self._train_callback,
                                                       eval_every_x_epochs=1,
                                                       max_unimproving_models=10,
                                                       custom_train_func=partial(
                                                           self.categorical_train_function,
                                                           test=False),
                                                       custom_test_func=partial(
                                                           self.categorical_train_function, test=True)
                                                       )

            self._model = best_model.to(self.device)

        elif all([x['output_type'] == COLUMN_DATA_TYPES.NUMERIC or x['output_type'] == COLUMN_DATA_TYPES.CATEGORICAL
                  for x in training_data['targets']]):

            self.desired_error = 0.01
            self._model_type = 'generic_target_predictor'
            self._model = self._embeddings_model_class.from_pretrained(self._pretrained_model_name).to(self.device)
            batch_size = 10
            self._max_len = min(max([len(x) for x in priming_data]), self._model.config.max_position_embeddings)

            self._head = DefaultNet(dynamic_parameters={}, shape=funnel(
                768, sum([len(x['encoded_output'][0]) for x in training_data['targets']]), depth=5))

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self._head.named_parameters() if not any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.000001},
                {'params': [p for n, p in self._head.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
            # optimizer = Ranger(self._head.parameters(),lr=5e-5)

            # num_training_steps is kind of an estimation
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=10, num_training_steps=len(priming_data) * 15 / 20)

            criterion = torch.nn.MSELoss()

            gym = Gym(model=self._head, optimizer=optimizer, scheduler=scheduler,
                      loss_criterion=criterion, device=self.device, name=self.name)

            input = [self._tokenizer.encode(x[:self._max_len], add_special_tokens=True) for x in priming_data]
            tokenized_max_len = max([len(x) for x in input])
            input = torch.tensor([x + [self._pad_id] * (tokenized_max_len - len(x)) for x in input])

            real = [[]] * len(training_data['targets'][0]['encoded_output'])
            for i in range(len(real)):
                for target in training_data['targets']:
                    real[i] = real[i] + list(target['encoded_output'][i])
            real = torch.tensor(real)

            merged_data = list(zip(input, real))

            train_data_loader = DataLoader(
                merged_data[:int(len(merged_data) * 9 / 10)], batch_size=batch_size, shuffle=True)
            test_data_loader = DataLoader(
                merged_data[int(len(merged_data) * 9 / 10):], batch_size=batch_size, shuffle=True)

            best_model, error, training_time = gym.fit(train_data_loader=train_data_loader,
                                                       test_data_loader=test_data_loader,
                                                       desired_error=self.desired_error,
                                                       max_time=self.max_training_time,
                                                       callback=self._train_callback,
                                                       eval_every_x_epochs=1,
                                                       max_unimproving_models=10,
                                                       custom_train_func=partial(
                                                           self.numerical_train_function,
                                                           backbone=self._model,
                                                           test=False),
                                                       custom_test_func=partial(
                                                           self.numerical_train_function,
                                                           backbone=self._model,
                                                           test=True)
                                                       )

            self._head = best_model.to(self.device)

        else:
            self._model_type = 'embeddings_generator'
            self._model = self._embeddings_model_class.from_pretrained(self._pretrained_model_name).to(self.device)

        self._prepared = True

    def encode(self, column_data):
        encoded_representation = []
        self._model = self._model.eval()
        with torch.no_grad():
            for text in column_data:
                if text is None:
                    text = ''
                input = torch.tensor(self._tokenizer.encode(
                    text[:self._max_len], add_special_tokens=True)).to(self.device).unsqueeze(0)

                if self._model_type == 'generic_target_predictor':
                    embeddings = self._model(input)
                    output = self._head(embeddings[0][:, 0, :])
                    encoded_representation.append(output.tolist()[0])

                elif self._model_type == 'classifier':
                    output = self._model(input)
                    logits = output[0]
                    predicted_targets = logits[0].tolist()
                    encoded_representation.append(predicted_targets)

                else:
                    output = self._model(input)
                    embeddings = output[0][:, 0, :].cpu().numpy()[0]
                    encoded_representation.append(embeddings)

        return torch.Tensor(encoded_representation)

    def decode(self, encoded_values_tensor, max_length=100):
        raise Exception('This encoder is not bi-directional')
