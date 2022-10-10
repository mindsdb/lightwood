import unittest
from lightwood.encoder.categorical import CategoricalAutoEncoder

import string
import random
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch

from lightwood.helpers.log import log


class TestAutoencoder(unittest.TestCase):

    def create_test_data(self,
                         nb_categories=500,
                         nb_int_categories=50,
                         max_category_size=50,
                         test_data_rel_size=0.33):
        random.seed(2)
        np_random = np.random.default_rng(seed=2)
        int_categories = np_random.integers(low=1, high=20, size=nb_int_categories)
        str_categories = [
            ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(7, 8)))
            for category_i in range(nb_categories - nb_int_categories)
        ]
        categories = list(int_categories) + str_categories
        category_sizes = np_random.integers(low=1, high=max_category_size, size=nb_int_categories)
        category_indexes = np.array(range(nb_categories), dtype=int)
        sample_category_indexes = np.repeat(category_indexes, category_sizes)
        np_random.shuffle(sample_category_indexes)
        samples = [ categories[i] for i in sample_category_indexes ]
        data_size = len(samples)
        priming_data = samples
        test_data = []
        if(test_data_rel_size > 0.):
            test_data_size = round(data_size * test_data_rel_size) + 1
            test_data = priming_data[:test_data_size]
        return priming_data, test_data

    def create_test_data_old(self):
        random.seed(2)
        cateogries = [''.join(random.choices(string.ascii_uppercase + string.digits,
                              k=random.randint(7, 8))) for x in range(500)]
        for i in range(len(cateogries)):
            if i % 10 == 0:
                cateogries[i] = random.randint(1, 20)

        priming_data = []
        test_data = []
        for category in cateogries:
            times = random.randint(1, 50)
            for i in range(times):
                priming_data.append(category)
                if i % 3 == 0 or i == 1:
                    test_data.append(category)

        random.shuffle(priming_data)
        random.shuffle(test_data)
        return priming_data, test_data

    def test_autoencoder(self):
        """
        Checks reconstruction accuracy above 70% for a set of categories, length 8, for up to 500 unique categories (actual around 468).
        """  # noqa
        log.setLevel(logging.DEBUG)

        torch.manual_seed(2)

        priming_data, test_data = self.create_test_data()

        enc = CategoricalAutoEncoder(stop_after=20)
        enc.desired_error = 3

        enc.prepare(pd.Series(priming_data), pd.Series(priming_data))
        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        encoder_accuracy = accuracy_score(list(map(str, test_data)), list(map(str, decoded_data)))
        print(f'Categorical encoder accuracy for: {encoder_accuracy} on testing dataset')
        self.assertTrue(encoder_accuracy > 0.70)

    def check_encoder_on_device(self, device):
        priming_data, _ = self.create_test_data(nb_categories=8,
                                                nb_int_categories=3,
                                                max_category_size=3,
                                                test_data_rel_size=0.)

        enc = CategoricalAutoEncoder(stop_after=5, device=device)
        enc.prepare(pd.Series(priming_data), pd.Series(priming_data))
        self.assertEqual(list(enc.net.parameters())[0].device.type, device)

    def test_encoder_on_cpu(self):
        self.check_encoder_on_device('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA unavailable')
    def test_encoder_on_cuda(self):
        self.check_encoder_on_device('cuda')
