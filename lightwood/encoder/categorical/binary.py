import torch
import numpy as np
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.constants import _UNCOMMON_WORD

from typing import Dict, List
from pandas import Series


class BinaryEncoder(BaseEncoder):
    def __init__(
        self,
        is_target: bool = False,
        target_class_distribution: Dict[str, float] = None,
    ):
        super().__init__(is_target)
        """
        Creates a 2-element vector representing categories :math:`A` and :math:`B` as such: 

        .. math::

           A &= [1, 0] \\
           B &= [0, 1]

        This encoder is a specialized case of one-hot encoding (OHE); unknown categories are explicitly handled as [0, 0].

        When data is typed with Lightwood, this class is only deployed if an input data type is explicitly recognized as binary (i.e. the column has only 2 unique values like True/False). If future data shows a new category (thus the data is no longer truly binary), this encoder will no longer be appropriate unless you are comfortable mapping ALL new classes as [0, 0].

        An encoder can also represent the target column; in this case, `is_target` is `True`, and `target_class_distribution`, from the `StatisticalAnalysis` phase. The `target_class_distribution` provides the relative percentage of each class in the data which is important for imbalanced populations.

        :param is_target: Whether encoder featurizes target column
        :param target_class_distribution: Percentage of total population represented by each category (from [0, 1]).
        """ # noqa

        self.map = {}  # category name -> index
        self.rev_map = {}  # index -> category name
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
        """ # noqa
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        unq_cats = np.unique([i for i in priming_data if i is not None]).tolist() 
        self.map = {cat: indx for indx, cat in enumerate(unq_cats)}
        self.rev_map = {indx: cat for cat, indx in self.map.items()}

        # Enforce only binary; map must have exactly 2 classes.
        if len(self.map) != 2:
            raise ValueError('Issue with dtype; data has > 2 classes.')

        # For target-only, report on relative weights of classes
        if self.is_target:
            self.index_weights = torch.Tensor([1, 1])  # Equally wt. both classes

            # If imbalanced detected, re-weight by inverse
            if self.target_class_distribution is not None:
                for cat in self.map.keys():
                    self.index_weights[self.map[cat]] = (
                        1 / self.target_class_distribution[cat]
                    )

        self.is_prepared = True

    def encode(self, column_data) -> torch.Tensor:
        """
        Encodes categories as OHE binary. Unknown/unrecognized classes return [0,0].

        :param column_data: Pre-processed data to encode
        :returns Encoded data of form :math:`N_{rows} x 2`
        """ # noqa
        if not self.is_prepared:
            raise Exception(
                'You need to call "prepare" before calling "encode" or "decode".'
            )

        ret = torch.zeros(size=(len(column_data), 2))

        for idx, word in enumerate(column_data):
            index = self.map.get(word, None)

            if index is not None:
                ret[idx, index] = 1

        return torch.Tensor(ret)

    def decode(self, encoded_data: torch.Tensor):
        """
        Given encoded data, return in form of original category labels.
        The input to `decode` makes no presumption on whether the data is already in OHE form OR not, as it some models may output a set of probabilities of weights assigned to each class. The decoded value will always be the argmax of such a vector.

        In the case that the vector is all 0s, the output is decoded as "UNKNOWN"

        :param encoded_data: the output of a mixer model

        :returns Decoded values for each data point
        """ # noqa
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            if not np.any(vector):  # Vector of all 0s -> unknown category
                ret.append(_UNCOMMON_WORD)
            else:
                ret.append(self.rev_map[np.argmax(vector)])

        return ret

    def decode_probabilities(self, encoded_data: torch.Tensor):
        """
        Provides decoded answers, as well as a probability assignment to each data point.

        :param encoded_data: the output of a mixer model

        :returns Decoded values for each data point, Probability vector for each category, and the reverse map of dimension to category name
        """ # noqa
        encoded_data_list = encoded_data.tolist()
        ret = []
        probs = []

        for vector in encoded_data_list:
            if not np.any(vector):  # Vector of all 0s -> unknown category
                ret.append(_UNCOMMON_WORD)
            else:
                ret.append(self.rev_map[np.argmax(vector)])

            probs.append(self._norm_vec(vector))

        return ret, probs, self.rev_map

    @staticmethod
    def _norm_vec(vec: List[float]):
        """
        Given a vector, normalizes so that the sum of elements is 1.

        :param vec: Assigned weights for each category
        """ # noqa
        total = sum(vec)
        return [i / total for i in vec]
