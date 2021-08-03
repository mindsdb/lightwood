import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightwood.model.helpers.ranger import Ranger
from lightwood.encoder.categorical.onehot import OneHotEncoder
from lightwood.encoder.categorical.gym import Gym
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.model.helpers.default_net import DefaultNet


class CategoricalAutoEncoder(BaseEncoder):
    def __init__(self, stop_after: int, is_target=False, max_encoded_length=100):
        super().__init__(is_target)
        self._prepared = False
        self.name = 'Categorical Autoencoder'
        self.net = None
        self.encoder = None
        self.decoder = None
        self.onehot_encoder = OneHotEncoder(is_target=self.is_target)
        self.desired_error = 0.01
        self.use_autoencoder = None
        self.max_encoded_length = max_encoded_length
        self.stop_after = stop_after
        # @TODO stop using instead of ONEHOT !!!@!
        self.is_nn_encoder = True
        self.output_size = None

    def _train_callback(self, error, real_buff, predicted_buff):
        log.info(f'{self.name} reached a loss of {error} while training !')

    def _encoder_targets(self, data):
        oh_encoded_categories = self.onehot_encoder.encode(data)
        target = oh_encoded_categories.cpu().numpy()
        target_indexes = np.where(target > 0)[1]
        targets_c = torch.LongTensor(target_indexes)
        labels = targets_c.to(self.net.device)
        return labels

    def prepare(self, priming_data):
        random.seed(len(priming_data))

        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.onehot_encoder.prepare(priming_data)

        input_len = self.onehot_encoder._lang.n_words
        self.use_autoencoder = self.max_encoded_length is not None and input_len > self.max_encoded_length

        if self.use_autoencoder:
            log.info('Preparing a categorical autoencoder, this might take a while')

            embeddings_layer_len = self.max_encoded_length

            self.net = DefaultNet(shape=[
                                  input_len, embeddings_layer_len, input_len])

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = Ranger(self.net.parameters())

            gym = Gym(model=self.net, optimizer=optimizer, scheduler=None, loss_criterion=criterion,
                      device=self.net.device, name=self.name, input_encoder=self.onehot_encoder.encode,
                      output_encoder=self._encoder_targets)

            batch_size = min(200, int(len(priming_data) / 50))

            priming_data_str = [str(x) for x in priming_data]
            train_data_loader = DataLoader(list(zip(priming_data_str,priming_data_str)), batch_size=batch_size, shuffle=True)

            test_data_loader = None

            best_model, error, training_time = gym.fit(train_data_loader,
                                                       test_data_loader,
                                                       desired_error=self.desired_error,
                                                       max_time=self.stop_after,
                                                       callback=self._train_callback,
                                                       eval_every_x_epochs=1,
                                                       max_unimproving_models=5)

            self.net = best_model.to(self.net.device)

            modules = [module for module in self.net.modules() if type(
                module) != torch.nn.Sequential and type(module) != DefaultNet]
            self.encoder = torch.nn.Sequential(*modules[0:2]).eval()
            self.decoder = torch.nn.Sequential(*modules[2:3]).eval()
            log.info('Categorical autoencoder ready')

        self.output_size = self.onehot_encoder._lang.n_words
        self._prepared = True

    def encode(self, column_data):
        oh_encoded_tensor = self.onehot_encoder.encode(column_data)
        if not self.use_autoencoder:
            return oh_encoded_tensor
        else:
            with torch.no_grad():
                oh_encoded_tensor = oh_encoded_tensor.to(self.net.device)
                embeddings = self.encoder(oh_encoded_tensor)
                return embeddings.to('cpu')

    def decode(self, encoded_data):
        if not self.use_autoencoder:
            return self.onehot_encoder.decode(encoded_data)
        else:
            with torch.no_grad():
                oh_encoded_tensor = self.decoder(encoded_data)
                oh_encoded_tensor = oh_encoded_tensor.to('cpu')
                return self.onehot_encoder.decode(oh_encoded_tensor)
