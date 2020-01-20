import math
import logging
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.encoders.categorical.onehot import OneHotEncoder
from lightwood.api.gym import Gym
from lightwood.config.config import CONFIG


MAX_LENGTH = 100

class CategoricalAutoEncoder:

    def __init__(self, is_target=False):
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self.name = 'Categorical Autoencoder'
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
        self.max_training_time = CONFIG.MAX_ENCODER_TRAINING_TIME

    def _train_callback(self, error, real_buff, predicted_buff):
        logging.info(f'{self.name} reached a loss of {error} while training !')

    def _encoder_targets(self, data):
        oh_encoded_categories = self.onehot_encoder.encode(data)
        target = oh_encoded_categories.cpu().numpy()
        target_indexes = np.where(target>0)[1]
        targets_c = torch.LongTensor(target_indexes)
        labels = targets_c.to(self.net.device)
        return labels

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

            gym = Gym(model=self.net, optimizer=optimizer, scheduler=None, loss_criterion=criterion, device=self.net.device, name=self.name, input_encoder=self.onehot_encoder.encode, output_encoder=self._encoder_targets)

            batch_size = min(200, int(len(priming_data)/50))

            priming_data_str = [str(x) for x in priming_data]
            train_data_loader = DataLoader(list(zip(priming_data_str,priming_data_str)), batch_size=batch_size, shuffle=True)
            test_data_loader = None

            best_model, error, training_time = gym.fit(train_data_loader, test_data_loader, desired_error=self.desired_error, max_time=self.max_training_time, callback=self._train_callback, eval_every_x_epochs=1, max_unimproving_models=5)

            self.net = best_model.to(self.net.device)

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
    import logging

    logging.getLogger().setLevel(logging.DEBUG)

    random.seed(2)
    cateogries = [''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(7,8))) for x in range(2000)]
    for i in range(len(cateogries)):
        if i % 10 == 0:
            cateogries[i] = random.randint(1,20)

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

    encoder_accuracy = accuracy_score(list(map(str,test_data)), list(map(str,decoded_data)))
    print(f'Categorical encoder accuracy for: {encoder_accuracy} on testing dataset')
    assert(encoder_accuracy > 0.98)
