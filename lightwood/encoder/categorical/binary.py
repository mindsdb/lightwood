from copy import deepcopy as dc
from typing import Dict, List, Iterable, Tuple

import torch
import numpy as np
from scipy.special import softmax

from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.constants import _UNCOMMON_WORD


class BinaryEncoder(BaseEncoder):
    """
   Creates a one-hot-encoding for binary class data. Assume two arbitrary categories :math:`A` and :math:`B`; representation for them will be as such:

    .. math::

       A &= [1, 0] \\
       B &= [0, 1]

    This encoder is a specialized case of one-hot encoding (OHE); unknown categories are explicitly handled as [0, 0]. Unknowns may only be reported if the input row value is NULL (or python `None` type) or if new data, after the encoder is prepared, has examples outside the feature map.

    When data is typed with Lightwood, this class is only deployed if an input data type is explicitly recognized as binary (i.e. the column has only 2 unique values like True/False). If future data shows a new category (thus the data is no longer truly binary), this encoder will no longer be appropriate unless you are comfortable mapping ALL new classes as [0, 0].

    An encoder can represent a feature column or target column; in this case it represents a target, `is_target` is `True`, and `target_weights`. The `target_weights` parameter enables users to specify how heavily each class should be weighted within a mixer - useful in imbalanced classes. 

    By default, `dataprep_ml.StatisticalAnalysis` phase will provide `target_weights` as the relative fraction of each class in the data which is important for imbalanced populations; for example, suppose there is a 80/20 imbalanced representation across 3 different classes - `target_weights` will be a vector as such::

    target_weights = {"class1": 0.8, "class2": 0.2}

    Users should note that models will be presented with the inverse of the target weights, `inv_target_weights`, which will perform the 1/target_value_per_class operation. **This means large values will result in small weights for the model**.
    """ # noqa
    def __init__(
        self,
        is_target: bool = False,
        target_weights: Dict[str, float] = None,
    ):
        super().__init__(is_target)
        """
        :param is_target: Whether encoder featurizes target column
        :param target_weights: Percentage of total population represented by each category (from [0, 1]), as a dictionary.
        """  # noqa

        self.map = {}  # category name -> index
        self.rev_map = {}  # index -> category name
        self.output_size = 2
        self.encoder_class_type = str

        # Weight-balance info if encoder represents target
        self.target_weights = None
        self.index_weights = None
        if self.is_target:
            self.target_weights = dc(target_weights)

    def prepare(self, priming_data: Iterable[str]):
        """
        Given priming data, create a map/inverse-map corresponding category name to index (and vice versa).

        :param priming_data: Binary data to encode
        """  # noqa
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        unq_cats = np.unique([i for i in priming_data if i is not None]).tolist()

        self.map = {cat: indx for indx, cat in enumerate(unq_cats)}
        self.rev_map = {indx: cat for cat, indx in self.map.items()}

        # Enforce only binary; map must have exactly 2 classes.
        if len(self.map) > 2:
            raise ValueError(f'Issue with dtype; data has > 2 classes. All classes are: {self.map}')

        # For target-only, report on relative weights of classes
        if self.is_target:

            self.index_weights = torch.Tensor([1, 1])  # Equally wt. both classes

            # If target weights provided, weight by inverse
            if self.target_weights is not None:

                # Check target weights properly specified
                if sum([np.isclose(i, 0) for i in self.target_weights.values()]) > 0:
                    raise ValueError('Target weights cannot be 0')

                # Ensure all classes are specified in the weights criteria
                assert(set(self.target_weights.keys()) == set(self.map.keys()))

                for cat in self.map.keys():
                    self.index_weights[self.map[cat]] = self.target_weights[cat]

        self.is_prepared = True

    def encode(self, column_data: Iterable[str]) -> torch.Tensor:
        """
        Encodes categories as OHE binary. Unknown/unrecognized classes return [0,0].

        :param column_data: Pre-processed data to encode
        :returns Encoded data of form :math:`N_{rows} x 2`
        """  # noqa
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

        :returns: Decoded values for each data point
        """  # noqa
        encoded_data_list = encoded_data.tolist()
        ret = []

        for vector in encoded_data_list:
            if not np.any(vector):  # Vector of all 0s -> unknown category
                ret.append(_UNCOMMON_WORD)
            else:
                ret.append(self.rev_map[np.argmax(vector)])

        return ret

    def decode_probabilities(self, encoded_data: torch.Tensor) -> Tuple[List[str], List[List[float]], Dict[int, str]]:
        """
        Provides decoded answers, as well as a probability assignment to each data point.

        :param encoded_data: the output of a mixer model

        :returns: Decoded values for each data point, Probability vector for each category, and the reverse map of dimension to category name
        """  # noqa
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
        Given a vector, normalizes so that the sum of elements is 1, using softmax.

        :param vec: Assigned weights for each category
        """  # noqa
        return softmax(vec)
