import torch
import numpy as np

from lightwood.mixers.helpers.default_net import DefaultNet
from lightwood.mixers.helpers.transformer import Transformer
from lightwood.mixers.helpers.ranger import Ranger
from lightwood.encoders.categorical.categorical import CategoricalEncoder


UNCOMMON_WORD = '<UNCOMMON>'
UNCOMMON_TOKEN = 0
MAX_LENGTH = 100

class AutoEncoder:

    def __init__(self, is_target = False):
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False
        self.net = None
        self.oh_encoder = CategoricalEncoder()

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self.oh_encoder.prepare_encoder(priming_data)

        input_len = self.oh_encoder._lang.n_words
        embeddings_layer_len = min(input_len/2,MAX_LENGTH)

        self.net = DefaultNet(dynamic_parameters={},shape=[input_len, embeddings_layer_len, input_len])

        for category in priming_data:
            encoded_category = self.oh_encoder(category)

        ds = torch.utils.data.TensorDataset(priming_data)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=200, shuffle=True)

        criterion = torch.nn.MSELoss()
        optimizer = Ranger(self.net.parameters())

        for epcohs in range(100):
            running_loss = 0
            error = 0
            for i, data in enumerate(data_loader, 0):
                oh_encoded_categories = data
                oh_encoded_categories = oh_encoded_categories.to(self.net.device)
                self.net(oh_encoded_categories)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                error = running_loss / (i + 1)
                print(error)

            if error < 0.02:
                break

        self._prepared = True

    def encode(self, column_data):
        oh_encoded_tensor = self.oh_encoder.encode(column_data)
        oh_encoded_tensor = oh_encoded_tensor.to(self.nn.device)
        embeddings = self.nn.modules()[0:3](oh_encoded_tensor)

        return embeddings


    def decode(self, encoded_data):
        oh_encoded_tensor = nn.modules()[1:4](encoded_data)
        oh_encoded_tensor = oh_encoded_tensor.to('cpu')
        decoded_categories = self.oh_encoder.decode(oh_encoded_tensor)

        return decoded_categories


if __name__ == "__main__":

    data = ['category 1', 'category 3', 'category 4', None]

    enc = CategoricalEncoder()

    enc.fit(data)
    encoded_data = enc.encode(data)
    decoded_data = enc.decode(enc.encode(['category 2', 'category 1', 'category 3', None]))

    assert(len(encoded_data) == 4)
    assert(decoded_data[1] == 'category 1')
    assert(decoded_data[2] == 'category 3')
    for i in [0,3]:
        assert(encoded_data[0][i] == UNCOMMON_TOKEN)
        assert(decoded_data[i] == UNCOMMON_WORD)

    print(f'Encoded values: \n{encoded_data}')
    print(f'Decoded values: \n{decoded_data}')
