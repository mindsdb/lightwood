import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightwood.mixer.helpers.ranger import Ranger
from lightwood.encoder.categorical.simple_label import SimpleLabelEncoder
from lightwood.encoder.categorical.onehot import OneHotEncoder
from lightwood.encoder.categorical.gym import Gym
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log
from lightwood.mixer.helpers.default_net import DefaultNet
import pandas as pd

from typing import Iterable, Tuple, List


class CategoricalAutoEncoder(BaseEncoder):
    """
    Trains an autoencoder (AE) to represent categorical information with over 100 categories. This is used to ensure that feature vectors for categorical data with many categories are not excessively large.

    The AE defaults to a vector sized 100 but can be adjusted to user preference. It is highly advised NOT to use this encoder to feature engineer your target, as reconstruction accuracy will determine your AE's ability to decode properly.

    """  # noqa

    is_trainable_encoder: bool = True

    def __init__(
            self,
            stop_after: float = 3600,
            is_target: bool = False,
            max_encoded_length: int = 100,
            desired_error: float = 0.01,
            batch_size: int = 200,
            device: str = '',
            input_encoder: str = None
    ):
        """
        :param stop_after: Stops training with provided time limit (sec)
        :param is_target: Encoder represents target class (NOT recommended)
        :param max_encoded_length: Maximum length of vector represented
        :param desired_error: Threshold for reconstruction accuracy error
        :param batch_size: Minimum batch size while training
        :param device: Name of the device that get_device_from_name will attempt to use
        :param input_encoder: one of `OneHotEncoder` or `SimpleLabelEncoder` to force usage of the underlying input encoder. Note that OHE does not scale for categorical features with high cardinality, while SLE can but is less accurate overall.
        """  # noqa
        super().__init__(is_target)
        self.is_prepared = False
        self.name = 'Categorical Autoencoder'
        self.output_size = max_encoded_length

        # Model details
        self.net = None
        self.encoder = None
        self.decoder = None
        self.input_encoder = None  # TBD at prepare()
        self.device_type = device
        self.input_encoder = input_encoder

        # Training details
        self.batch_size = batch_size
        self.desired_error = desired_error
        self.stop_after = stop_after

    def prepare(self, train_priming_data: pd.Series, dev_priming_data: pd.Series):
        """
        Creates inputs and prepares a categorical autoencoder (CatAE) for input data. Currently, does not support a dev set; inputs for train and dev are concatenated together to train an autoencoder.

        :param train_priming_data: Input training data
        :param dev_priming_data: Input dev data (Not supported currently)
        """  # noqa

        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if self.is_target:
            log.warning('You are trying to use an autoencoder for the target value! This is very likely a bad idea.')

        error_msg = f'Provided an invalid input encoder ({self.input_encoder}), please use either `OneHotEncoder` or `SimpleLabelEncoder`.'  # noqa
        if self.input_encoder is not None:
            assert self.input_encoder in ('OneHotEncoder', 'SimpleLabelEncoder'), error_msg

        log.info('Preparing a categorical autoencoder.')

        if self.input_encoder == 'SimpleLabelEncoder' or \
                (self.input_encoder is None and train_priming_data.nunique() > 500):
            log.info('Deploying SimpleLabelEncoder for CategoricalAutoEncoder input.')
            self.input_encoder = SimpleLabelEncoder(is_target=self.is_target)
            input_len = self.input_encoder.output_size
            self.output_size = 32
            net_shape = [input_len, 128, 64, self.output_size, input_len]
        else:
            log.info('Deploying OneHotEncoder for CategoricalAutoEncoder input.')
            self.input_encoder = OneHotEncoder(is_target=self.is_target)
            net_shape = None   # defined at prepare() due to the OHE output size being determined then

        train_loader, dev_loader = self._prepare_AE_input(
            train_priming_data, dev_priming_data
        )

        best_model = self._prepare_catae(train_loader, dev_loader, net_shape=net_shape)
        self.net = best_model.to(self.net.device)

        modules = [
            module
            for module in self.net.modules()
            if type(module) != torch.nn.Sequential and type(module) != DefaultNet
        ]

        self.encoder = torch.nn.Sequential(*modules[0:-1]).eval()
        self.decoder = torch.nn.Sequential(*modules[-1:]).eval()
        log.info('Categorical autoencoder ready.')

        self.is_prepared = True

    def encode(self, column_data: Iterable[str]) -> torch.Tensor:
        """
        Encodes categorical information in column as the compressed vector from the CatAE.

        :param column_data: An iterable of category samples from a column

        :returns: An embedding for each sample in original input
        """  # noqa
        encoded_tensor = self.input_encoder.encode(column_data)

        with torch.no_grad():
            encoded_tensor = encoded_tensor.to(self.net.device)
            if len(encoded_tensor.shape) < 2:
                encoded_tensor = encoded_tensor.unsqueeze(-1)
            embeddings = self.encoder(encoded_tensor)
            return embeddings.to('cpu')

    def decode(self, encoded_data: torch.Tensor) -> List[str]:
        """
        Decodes from the embedding space, the original categories.

        ..warning If your reconstruction accuracy is not 100%, the CatAE may not return the correct category.

        :param encoded_data: A torch tensor of embeddings for category predictions

        :returns: A list of 'translated' categories for each embedding
        """  # noqa
        with torch.no_grad():
            encoded_data = encoded_data.to(self.net.device)
            encoded_tensor = self.decoder(encoded_data)
            encoded_tensor = encoded_tensor.to('cpu')
            return self.input_encoder.decode(encoded_tensor)

    def _prepare_AE_input(
            self, train_priming_data: pd.Series, dev_priming_data: pd.Series
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Creates the data loaders for the CatAE model inputs. Expected inputs are generally of form `pd.Series`

        Currently does not use 'dev'; concatenates both inputs together.

        Input to the `DataLoader` must be an Iterable[str] (ideally List[str])

        """  # noqa
        if len(dev_priming_data) > 0:
            priming_data = (
                pd.concat([train_priming_data, dev_priming_data]).astype(str).tolist()
            )
        else:
            priming_data = [str(x) for x in train_priming_data]

        random.seed(len(priming_data))

        # Prepare a one-hot encoder for CatAE inputs
        self.input_encoder.prepare(priming_data)
        self.batch_size = max(min(self.batch_size, int(len(priming_data) / 50)), 1)

        train_loader = DataLoader(
            list(zip(priming_data, priming_data)),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # TODO; make `Gym` compatible with a dev set
        dev_loader = None

        return train_loader, dev_loader

    def _prepare_catae(self, train_loader: DataLoader, dev_loader: DataLoader, net_shape=None):
        """
        Trains the CatAE using Lightwood's `Gym` class.

        :param train_loader: Training dataset Loader
        :param dev_loader: Validation set DataLoader
        """  # noqa
        if net_shape is None:
            input_len = self.input_encoder.output_size
            net_shape = [input_len, self.output_size, input_len]

        self.net = DefaultNet(shape=net_shape, device=self.device_type)

        if isinstance(self.input_encoder, OneHotEncoder):
            criterion = torch.nn.CrossEntropyLoss()
            desired_error = self.desired_error
        elif isinstance(self.input_encoder, SimpleLabelEncoder):
            criterion = torch.nn.MSELoss()
            desired_error = 1e-9
        else:
            raise Exception(f'[CatAutoEncoder] Input encoder of type {type(self.input_encoder)} is not supported!')

        if isinstance(self.input_encoder, OneHotEncoder):
            optimizer = Ranger(self.net.parameters())
            output_encoder = self._encoder_targets
            max_time = self.stop_after
        else:
            optimizer = Ranger(self.net.parameters(), weight_decay=1e-2)
            output_encoder = self._label_targets
            max_time = 60 * 2

        gym = Gym(
            model=self.net,
            optimizer=optimizer,
            scheduler=None,
            loss_criterion=criterion,
            device=self.net.device,
            name=self.name,
            input_encoder=self.input_encoder.encode,
            output_encoder=output_encoder,
        )

        best_model, _, _ = gym.fit(
            train_loader,
            dev_loader,
            desired_error=desired_error,
            max_time=max_time,
            eval_every_x_epochs=1,
            max_unimproving_models=5,
        )

        return best_model

    def _encoder_targets(self, data):
        """ Encodes target data with a OHE encoder """
        encoded_categories = self.input_encoder.encode(data)
        target = encoded_categories.cpu().numpy()
        target_indexes = np.where(target > 0)[1]
        targets_c = torch.LongTensor(target_indexes)
        labels = targets_c.to(self.net.device)
        return labels

    def _label_targets(self, data):
        """ Encodes target data with a label encoder """
        data = pd.Series(data)
        enc = self.input_encoder.encode(data)
        if len(enc.shape) < 2:
            enc = enc.unsqueeze(-1)
        return enc
