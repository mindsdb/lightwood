import time
import copy

import numpy as np
import torch
from pytorch_transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertTokenizer, AdamW

from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class DistilBertEncoder:
    def __init__(self, is_target=False):
        self._tokenizer = None
        self._model = None
        self._pad_token = None
        self._pytorch_wrapper = torch.FloatTensor
        self._max_len = None
        self._max_ele = None
        self._prepared = False
        self._model_type = None
        self.desired_error = 0.05
        self.max_training_time = 1800

        device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE
        self.device = torch.device(device_str)

    def prepare_encoder(self, priming_data, training_data=None):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._max_len = min(max([len(x) for x in priming_data]),768)
        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._pad_token = self._tokenizer.convert_tokens_to_ids([self._tokenizer])[0]
        # @TODO: Support multiple targets if they are all categorical or train for the categorical target if it's a mix (maybe ?)
        # @TODO: Attach a language modeling head and/or use GPT2 and/or provide outputs better suited to a LM head (which will be the mixer) if the output if text

        if training_data is not None and 'targets' in training_data and len(training_data['targets']) ==1 and training_data['targets'][0]['output_type'] == COLUMN_DATA_TYPES.CATEGORICAL and CONFIG.TRAIN_TO_PREDICT_TARGET:
            self._model_type = 'classifier'
            self._model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(training_data['targets'][0]['unencoded_output'])) + 1).to(self.device)

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self._model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0001},
                {'params': [p for n, p in self._model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
            # This import is present in github but not in the pypi library yet
            #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

            self._model.train()
            error_buffer = []
            started = time.time()

            best_error = None
            best_model = None
            best_epoch = None

            for epoch in range(5000):
                running_loss = 0
                error = None

                '''
                itterable_priming_data = zip(*[iter(priming_data)]*batch_size)
                for i, data in enumerate(itterable_priming_data):

                    self._pad_token
                '''
                for i in range(len(priming_data)):
                    input = torch.tensor(self._tokenizer.encode(priming_data[i][:self._max_len], add_special_tokens=True)).to(self.device).unsqueeze(0)

                    labels = torch.tensor([torch.argmax(training_data['targets'][0]['encoded_output'][i])]).to(self.device)

                    outputs = self._model(input, labels=labels)
                    loss, logits = outputs[:2]
                    loss.backward()
                    #if i % 24 == 0:
                    #    print(torch.argmax(logits[0]), '----', labels[0])

                    optimizer.step()
                    self._model.zero_grad()

                    running_loss += loss.item()
                    error = running_loss/(i + 1)

                print(f'Text encoder training error: {error}')
                if best_error is None or best_error > error:
                    best_error = error
                    # Move to CPU to save GPU memory, move back to the origianl device if we end up using it
                    best_model = copy.deepcopy(self._model).cpu()
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
            self._model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)

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
