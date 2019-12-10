import torch
import numpy as np
import math
import logging
import random

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.encoders.categorical.onehot import OneHotEncoder


MAX_LENGTH = 100

class CategoricalAutoEncoder:

    def __init__(self, is_target=False):
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self.net = None
        self.encoder = None
        self.decoder = None
        self.onehot_encoder = OneHotEncoder()
        self.desired_error = 0.01
        self.use_autoencoder = None
        if is_target:
            self.max_encoded_length = None
        else:
            self.max_encoded_length = 100


    def prepare_encoder(self, priming_data):
        random.seed(len(priming_data))

        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self.onehot_encoder.prepare_encoder(priming_data)

        input_len = self.onehot_encoder._lang.n_words
        self.use_autoencoder = self.max_encoded_length is not None and input_len > self.max_encoded_length
        if self.use_autoencoder:
            logging.info('Preparing a categorical autoencoder, this might take a while')

            embeddings_layer_len = self.max_encoded_length

            self.net = DefaultNet(ds=None, dynamic_parameters={},shape=[input_len, embeddings_layer_len, input_len], selfaware=False)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = Ranger(self.net.parameters())

            batch_size = min(200, int(len(priming_data)/50))

            error_buffer = []
            for epcohs in range(5000):
                running_loss = 0
                error = 0
                random.shuffle(priming_data)
                itterable_priming_data = zip(*[iter(priming_data)]*batch_size)

                for i, data in enumerate(itterable_priming_data):
                    oh_encoded_categories = self.onehot_encoder.encode(data)
                    oh_encoded_categories = oh_encoded_categories.to(self.net.device)
                    self.net(oh_encoded_categories)

                    optimizer.zero_grad()

                    outputs = self.net(oh_encoded_categories)

                    target = oh_encoded_categories.cpu().numpy()
                    target_indexes = np.where(target>0)[1]
                    targets_c = torch.LongTensor(target_indexes)
                    labels = targets_c.to(self.net.device)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    error = running_loss / (i + 1)
                    error_buffer.append(error)

                logging.info(f'Categorial autoencoder training error: {error}')
                if len(error_buffer) > 5:
                    error_buffer.append(error)
                    error_buffer = error_buffer[-5:]
                    delta_mean = np.mean(error_buffer)
                    if delta_mean < 0 or error < self.desired_error:
                        break

            modules = [module for module in self.net.modules() if type(module) != torch.nn.Sequential and type(module) != DefaultNet]
            self.encoder = torch.nn.Sequential(*modules[0:2])
            self.decoder = torch.nn.Sequential(*modules[2:3])
            logging.info('Categorical autoencoder ready')

        self._prepared = True

    def encode(self, column_data):
        oh_encoded_tensor = self.onehot_encoder.encode(column_data)
        if not self.use_autoencoder:
            return oh_encoded_tensor
        else:
            oh_encoded_tensor = oh_encoded_tensor.to(self.net.device)
            embeddings = self.encoder(oh_encoded_tensor)
            return embeddings


    def decode(self, encoded_data):
        if not self.use_autoencoder:
            return self.onehot_encoder.decode(encoded_data)
        else:
            oh_encoded_tensor = self.decoder(encoded_data)
            oh_encoded_tensor = oh_encoded_tensor.to('cpu')
            decoded_categories = self.onehot_encoder.decode(oh_encoded_tensor)
            return decoded_categories


if __name__ == "__main__":
    # Generate some tests data
    import random
    import string
    from sklearn.metrics import accuracy_score

    random.seed(2)
    cateogries = [''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) for x in range(2000)]

    priming_data = []
    test_data = []
    for category in cateogries:
        times = random.randint(1,50)
        for i in range(times):
            priming_data.append(category)
            if i % 3 == 0 or i == 1:
                test_data.append(category)

    random.shuffle(priming_data)
    random.shuffle(test_data)

    enc = CategoricalAutoEncoder()

    enc.prepare_encoder(priming_data)
    encoded_data = enc.encode(test_data)
    decoded_data = enc.decode(encoded_data)

    encoder_accuracy = accuracy_score(list(test_data), decoded_data)
    print(f'Categorial encoder accuracy for: {encoder_accuracy} on testing dataset')
    assert(encoder_accuracy > 0.98)
