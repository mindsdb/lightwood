"""
Creates a 2-element vector representing categories :math:`A` and :math:`B` as such: 

.. math::

   A &= [1, 0] \\
   B &= [0, 1]

This is a specialized case of one-hot encoding (OHE); this is to explicitly enforce *no* possibility of an unknown class, as our default OHE does. When data is typed with Lightwood, this class is only deployed if the type is explicitly considered binary (i.e. the column has no missing values, otherwise it's considered via categorical one-hot or autoencoder).

An encoder can also represent the target column; in this case, `is_target` is True, and `target_class_distribution`, from the `StatisticalAnalysis` phase. The `target_class_distribution` provides the relative percentage of each class in the data which is important for imbalanced populations. 
"""

import torch
import numpy as np
from scipy.special import softmax
from lightwood.encoder.base import BaseEncoder

from typing import Dict, List
from pandas import Series

class BinaryEncoder(BaseEncoder):

    def __init__(self, is_target: bool = False, target_class_distribution: Dict[str, float] = None):
        super().__init__(is_target)

        self.map = {} # category name -> index
        self.rev_map = {} # index -> category name
        self.output_size = 2

        # Weight-balance info if encoder represents target
        if self.is_target:
            self.target_class_distribution = target_class_distribution
            self.index_weights = None

    def prepare(self, priming_data: Series):
        """
        Given priming data, create a map/inverse-map corresponding category name to index (and vice versa).

        If encoder represents target, also instantiates `index_weights` which enables downstream models to weight classes.

        :param priming_data: Binary data to encode
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        # For each member in the series, map 
        # Enforce strings
        priming_data = priming_data.astype('str')
        self.map = {cat: indx for indx, cat in enumerate(priming_data.unique())}
        self.rev_map = {indx: cat for cat, indx in self.map.items()}

        # Enforce only binary details
        assert(len(self.map) == 2, 'Issue with dtype; data has > 2 classes.')

        # For target-only, report on relative weights of classes
        if self.is_target:
            self.index_weights = torch.Tensor([1, 1]) # Equally wt. both classes

            # If imbalanced detected, re-weight by inverse
            if self.target_class_distribution is not None:
                for cat in self.map.keys():
                    self.index_weights[self.map[cat]] = 1 / self.target_class_distribution[cat]

        self.is_prepared = True

    def encode(self, column_data) -> torch.Tensor:
        """
        Encodes categories as OHE binary. Unknown/unrecognized classes return [0,0].

        :param column_data: Pre-processed data to encode
        :returns Encoded data of form :math:`N_{rows} x 2`
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')
        ret = []

        for word in column_data:
            ret.append([0, 0])
            index = self.map.get(word, None)
            
            if index is not None:
                ret[-1][index] = 1

        return torch.Tensor(ret)

    def decode(self, encoded_data, return_raw=False):
        """
        Given encoded data, return in form of original category labels.
        """
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            ret.append(self.rev_map[np.argmax(vector)])

            if return_raw:
                probs.append(softmax(vector).tolist())

        if return_raw:
            return ret, probs, self.rev_map
        else:
            return ret
