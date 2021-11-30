import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightwood.mixer.helpers.ranger import Ranger
from lightwood.encoder.categorical.onehot import OneHotEncoder
from lightwood.encoder.categorical.gym import Gym
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.mixer.helpers.default_net import DefaultNet
import pandas as pd


class CategoricalAutoEncoder(BaseEncoder):
    is_trainable_encoder: bool = True

    def __init__(self, stop_after: float = 3600, is_target: bool = False, max_encoded_length: int = 100):
        super().__init__(is_target)
        self.is_prepared = False
        self.name = 'Categorical Autoencoder'
        self.net = None
        self.encoder = None
        self.decoder = None
        self.onehot_encoder = OneHotEncoder(is_target=self.is_target)
        self.desired_error = 0.01
        self.stop_after = stop_after
        # @TODO stop using instead of ONEHOT !!!@!
        self.output_size = None
        self.max_encoded_length = max_encoded_length

    def _encoder_targets(self, data):
        oh_encoded_categories = self.onehot_encoder.encode(data)
        target = oh_encoded_categories.cpu().numpy()
        target_indexes = np.where(target > 0)[1]
        targets_c = torch.LongTensor(target_indexes)
        labels = targets_c.to(self.net.device)
        return labels

    def prepare(self, train_priming_data, dev_priming_data):
        priming_data = pd.concat([train_priming_data, dev_priming_data])
        random.seed(len(priming_data))

        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.onehot_encoder.prepare(priming_data)

        input_len = self.onehot_encoder.output_size

        if self.is_target:
            log.warning('You are trying to use an autoencoder for the target value! \
            This is very likely a bad idea')
        log.info('Preparing a categorical autoencoder, this might take a while')

        embeddings_layer_len = self.max_encoded_length

        self.net = DefaultNet(shape=[input_len, embeddings_layer_len, input_len])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = Ranger(self.net.parameters())

        gym = Gym(model=self.net, optimizer=optimizer, scheduler=None, loss_criterion=criterion,
                  device=self.net.device, name=self.name, input_encoder=self.onehot_encoder.encode,
                  output_encoder=self._encoder_targets)

        batch_size = min(200, int(len(priming_data) / 50))

        priming_data_str = [str(x) for x in priming_data]
        train_data_loader = DataLoader(
            list(zip(priming_data_str, priming_data_str)),
            batch_size=batch_size, shuffle=True)

        test_data_loader = None

        best_model, _, _ = gym.fit(train_data_loader,
                                   test_data_loader,
                                   desired_error=self.desired_error,
                                   max_time=self.stop_after,
                                   eval_every_x_epochs=1,
                                   max_unimproving_models=5)

        self.net = best_model.to(self.net.device)

        modules = [module for module in self.net.modules() if type(
            module) != torch.nn.Sequential and type(module) != DefaultNet]
        self.encoder = torch.nn.Sequential(*modules[0:2]).eval()
        self.decoder = torch.nn.Sequential(*modules[2:3]).eval()
        log.info('Categorical autoencoder ready')

        self.output_size = self.onehot_encoder.output_size
        self.output_size = self.max_encoded_length
        self.is_prepared = True

    def encode(self, column_data):
        oh_encoded_tensor = self.onehot_encoder.encode(column_data)

        with torch.no_grad():
            oh_encoded_tensor = oh_encoded_tensor.to(self.net.device)
            embeddings = self.encoder(oh_encoded_tensor)
            return embeddings.to('cpu')

    def decode(self, encoded_data):
        with torch.no_grad():
            encoded_data = encoded_data.to(self.net.device)
            oh_encoded_tensor = self.decoder(encoded_data)
            oh_encoded_tensor = oh_encoded_tensor.to('cpu')
            return self.onehot_encoder.decode(oh_encoded_tensor)
