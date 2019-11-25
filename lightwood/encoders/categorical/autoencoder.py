import torch
import numpy as np
import math

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.encoders.categorical.categorical import CategoricalEncoder


UNCOMMON_WORD = '<UNCOMMON>'
UNCOMMON_TOKEN = 0
MAX_LENGTH = 100

class CategoricalAutoEncoder:

    def __init__(self, is_target = False):
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self.net = None
        self.encoder = None
        self.decoder = None
        self.oh_encoder = CategoricalEncoder()

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self.oh_encoder.prepare_encoder(priming_data)

        input_len = self.oh_encoder._lang.n_words
        embeddings_layer_len = min(int(math.ceil(input_len/2)),MAX_LENGTH)

        self.net = DefaultNet(ds=None, dynamic_parameters={},shape=[input_len, embeddings_layer_len, input_len])

        encoded_priming_data = self.oh_encoder.encode(priming_data)

        data_loader = torch.utils.data.DataLoader(encoded_priming_data, batch_size=200, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = Ranger(self.net.parameters())

        for epcohs in range(20):
            running_loss = 0
            error = 0
            for i, data in enumerate(data_loader, 0):
                oh_encoded_categories = data
                oh_encoded_categories = torch.Tensor(oh_encoded_categories)
                oh_encoded_categories = oh_encoded_categories.to(self.net.device)
                self.net(oh_encoded_categories)

                optimizer.zero_grad()

                outputs = self.net(oh_encoded_categories)
                loss = criterion(outputs, oh_encoded_categories)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                error = running_loss / (i + 1)
                print(error)

            if error < 0.005:
                break

        modules = [module for module in self.net.modules() if type(module) != torch.nn.Sequential and type(module) != DefaultNet]
        self.encoder = torch.nn.Sequential(*modules[0:2])
        self.decoder = torch.nn.Sequential(*modules[2:3])

        self._prepared = True

    def encode(self, column_data):
        oh_encoded_tensor = self.oh_encoder.encode(column_data)
        oh_encoded_tensor = oh_encoded_tensor.to(self.net.device)
        embeddings = self.encoder(oh_encoded_tensor)

        return embeddings


    def decode(self, encoded_data):
        oh_encoded_tensor = self.decoder(encoded_data)
        oh_encoded_tensor = oh_encoded_tensor.to('cpu')
        decoded_categories = self.oh_encoder.decode(oh_encoded_tensor)
        return decoded_categories


if __name__ == "__main__":
    # Generate some tests data
    import random
    import string
    import pandas as pd
    from sklearn.metrics import accuracy_score

    random.seed(2)
    cateogries = [''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) for x in range(1000)]

    priming_data = []
    test_data = []
    for category in cateogries:
        times = random.randint(1,600)
        for i in range(times):
            priming_data.append(category)
            if i % 3 == 0 or i == 1:
                test_data.append(category)

    priming_data = pd.Series(priming_data).sample(1, random_state=2)
    test_data = pd.Series(test_data).sample(1, random_state=2)
    priming_data = list(priming_data)
    test_data = list(test_data)
    
    enc = CategoricalAutoEncoder()

    enc.prepare_encoder(priming_data)
    encoded_data = enc.encode(test_data)
    decoded_data = enc.decode(test_data)

    print(accuracy_score(list(test_data), decoded_data))
