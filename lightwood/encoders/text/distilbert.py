import time
import copy
import random
import logging

import numpy as np
import torch
from transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertTokenizer, AlbertModel, AlbertForSequenceClassification, DistilBertTokenizer, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup

from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, ENCODER_AIM


class DistilBertEncoder:
    def __init__(self, is_target=False, aim=ENCODER_AIM.BALANCE):
        self._tokenizer = None
        self._model = None
        self._pad_id = None
        self._pytorch_wrapper = torch.FloatTensor
        self._max_len = None
        self._max_ele = None
        self._prepared = False
        self._model_type = None
        self.desired_error = 0.05
        self.max_training_time = CONFIG.MAX_ENCODER_TRAINING_TIME
        # Possible: speed, balance, accuracy
        self.aim = aim

        if self.aim == ENCODER_AIM.SPEED:
            # uses more memory, takes very long to train and outputs weird debugging statements to the command line, consider waiting until it gets better or try to investigate why this happens (changing the pretrained model doesn't seem to help)
            self._classifier_model_class = AlbertForSequenceClassification
            self._embeddings_model_class = AlbertModel
            self._tokenizer_class = AlbertTokenizer
            self._pretrained_model_name = 'albert-base-v2'
            self._model_max_len = 768
        if self.aim == ENCODER_AIM.BALANCE:
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizer
            self._pretrained_model_name = 'distilbert-base-uncased'
            self._model_max_len = 768
        if self.aim == ENCODER_AIM.ACCURACY:
            self._classifier_model_class = DistilBertForSequenceClassification
            self._embeddings_model_class = DistilBertModel
            self._tokenizer_class = DistilBertTokenizer
            self._pretrained_model_name = 'distilbert-base-uncased'
            self._model_max_len = 768

        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE
        self.device = torch.device(device_str)

    def prepare_encoder(self, priming_data, training_data=None):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        priming_data = [x if x is not None else '' for x in priming_data]
        
        self._max_len = min(max([len(x) for x in priming_data]),self._model_max_len)
        self._tokenizer = self._tokenizer_class.from_pretrained(self._pretrained_model_name)
        self._pad_id = self._tokenizer.convert_tokens_to_ids([self._tokenizer.pad_token])[0]
        # @TODO: Support multiple targets if they are all categorical or train for the categorical target if it's a mix (maybe ?)
        # @TODO: Attach a language modeling head and/or use GPT2 and/or provide outputs better suited to a LM head (which will be the mixer) if the output if text

        if training_data is not None and 'targets' in training_data and len(training_data['targets']) ==1 and training_data['targets'][0]['output_type'] == COLUMN_DATA_TYPES.CATEGORICAL and CONFIG.TRAIN_TO_PREDICT_TARGET:
            self._model_type = 'classifier'
            self._model = self._classifier_model_class.from_pretrained(self._pretrained_model_name, num_labels=len(set(training_data['targets'][0]['unencoded_output'])) + 1).to(self.device)

            if self.aim == ENCODER_AIM.SPEED:
                batch_size = 10
            else:
                batch_size = 10

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.000001},
                {'params': [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
            # num_training_steps is kind of an estimation
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=len(priming_data) * 15/20)

            self._model.train()
            error_buffer = []
            started = time.time()

            best_error = None
            best_model = None
            best_epoch = None

            random.seed(len(priming_data))

            for epoch in range(5000):
                running_loss = 0
                error = None

                merged_data = list(zip(priming_data, training_data['targets'][0]['encoded_output']))
                random.shuffle(merged_data)

                randomized_priming_data, randomized_target_data = zip(*merged_data)

                itterable_priming_data = zip(*[iter(randomized_priming_data)]*batch_size)

                for i, data in enumerate(itterable_priming_data):
                    inputs = []
                    for text in data:
                        input = self._tokenizer.encode(text[:self._max_len], add_special_tokens=True)
                        inputs.append(input)

                    max_input = max([len(x) for x in inputs])
                    inputs = [x + [self._pad_id] * (max_input - len(x)) for x in inputs]
                    inputs = torch.tensor(inputs).to(self.device)

                    labels = torch.tensor([torch.argmax(x) for x in randomized_target_data[i*batch_size:(i+1)*batch_size]]).to(self.device)

                    outputs = self._model(inputs, labels=labels)
                    loss, logits = outputs[:2]
                    loss.backward()
                    running_loss += loss.item()

                    optimizer.step()
                    scheduler.step()

                    self._model.zero_grad()

                    error = running_loss/(i + 1)

                    if i % 200 == 0:
                        logging.info(f'Intermediate text encoder error: {error}')

                logging.info(f'Text encoder training error: {error}')
                if best_error is None or best_error > error:
                    best_error = error
                    # Move to CPU to save GPU memory, move back to the origianl device if we end up using it
                    self._model = self._model.cpu()
                    best_model = copy.deepcopy(self._model)
                    self._model = self._model.to(self.device)
                    best_epoch = epoch

                error_buffer.append(error)

                if len(error_buffer) > 3:
                    error_buffer.append(error)
                    error_buffer = error_buffer[-3:]
                    delta_mean = np.mean(error_buffer)
                    if delta_mean < 0 or error < self.desired_error or best_epoch < epoch - 5:
                        self._model = best_model.to(self.device)
                        break

                if started + self.max_training_time < time.time():
                    self._model = best_model.to(self.device)
                    break
        else:
            self._model_type = 'embeddings_generator'
            self._model = self._embeddings_model_class.from_pretrained(self._pretrained_model_name).to(self.device)

        self._prepared = True


    def encode(self, column_data):
        encoded_representation = []
        self._model.eval()
        with torch.no_grad():
            for text in column_data:
                if text is None:
                    text = ''
                input = torch.tensor(self._tokenizer.encode(text[:self._max_len], add_special_tokens=True)).to(self.device).unsqueeze(0)
                output = self._model(input)

                if self._model_type == 'classifier':
                    logits = output[0]
                    predicted_targets = logits[0].tolist()
                    encoded_representation.append(predicted_targets)
                else:
                    embeddings = output[0][:,0,:].cpu().numpy()[0]
                    encoded_representation.append(embeddings)

        return self._pytorch_wrapper(encoded_representation)

    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass
