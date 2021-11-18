import torch
import numpy as np
from lightwood.helpers.log import log
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.constants import _UNCOMMON_WORD

from typing import Dict, List, Iterable


class OneHotEncoder(BaseEncoder):
    """
    Creates a one-hot encoding (OHE) of categorical data. This creates a vector where each individual dimension corresponds to a category. A category has a 1:1 mapping between dimension indicated by a "1" in that position.

    OHE operates in 2 modes:
        (1) "use_unknown=True": Makes an :math:`N+1` length vector for :math:`N` categories, the first index always corresponds to the unknown category.

        (2) "use_unknown=False": Makes an :math:`N` length vector for :math:`N` categories, where an empty vector of 0s indicates an unknown/missing category.

    An encoder can also represent the target column; in this case, `is_target` is `True`, and `target_weights`. The `target_weights` parameter enables users to specify how heavily each class should be weighted within a mixer - useful in imbalanced classes.

    By default, the `StatisticalAnalysis` phase will provide `target_weights` as the relative fraction of each class in the data which is important for imbalanced populations; for example, suppose there is a 80/10/10 imbalanced representation across 3 different classes - `target_weights` will be a vector as such::

    target_weights = {"class1": 0.9, "class2": 0.1, "class3": 0.1}

    Users should note that models will be presented with the inverse of the target weights, `inv_target_weights`, which will perform the 1/target_value_per_class operation. **This means large values will result in small weights for the model**.
    """ # noqa
    def __init__(
        self,
        is_target: bool = False,
        target_weights: Dict[str, float] = None,
        use_unknown: bool = True,
    ):
        """
        :param is_target: True if this encoder featurizes the target column
        :param target_weights: Percentage of total population represented by each category (between [0, 1]).
        :param mode: True uses an extra dimension to account for unknown/out-of-distribution categories
        """  # noqa

        super().__init__(is_target)
        self.map = None  # category name -> index
        self.rev_map = None  # index -> category name
        self.use_unknown = use_unknown

        self.target_weights = None
        self.inv_target_weights = None
        if self.is_target:
            self.target_weights = target_weights

    def prepare(self, priming_data: Iterable[str]):
        """
        Prepares the OHE Encoder by creating a dictionary mapping.

        Unknown categories must be explicitly handled as python `None` types.
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        unq_cats = np.unique([i for i in priming_data if i is not None]).tolist()
        if self.use_unknown:
            log.info("Encoding UNK categories as index 0")
            self.map = {cat: indx + 1 for indx, cat in enumerate(unq_cats)}
            self.map.update({_UNCOMMON_WORD: 0})
            self.rev_map = {indx: cat for cat, indx in self.map.items()}
        else:
            log.info("Encoding UNK categories as vector of all 0s")
            self.map = {cat: indx for indx, cat in enumerate(unq_cats)}
            self.rev_map = {indx: cat for cat, indx in self.map.items()}

        # Set the length of output
        self.output_size = len(self.map)

        # For target-only, report on relative weights of classes
        # Each dimension of the inv_target_weights respects `map`
        if self.is_target:

            # Equally wt. all classes
            self.inv_target_weights = torch.ones(size=(self.output_size,))

            # If imbalanced detected, weight by inverse
            if self.target_weights is not None:

                # Check target weights properly specified
                if sum([np.isclose(i, 0) for i in self.target_weights.values()]) > 0:
                    raise ValueError('Target weights cannot be 0')

                for cat in self.map.keys():

                    if cat != _UNCOMMON_WORD:
                        self.inv_target_weights[self.map[cat]] = (
                            1 / self.target_weights[cat]
                        )

                # If using an unknown category, set to smallest possible value
                if self.use_unknown:
                    self.inv_target_weights[0] = self.inv_target_weights.min().item()

        self.is_prepared = True

    def encode(self, column_data: Iterable[str]) -> torch.Tensor:
        """
        Encodes pre-processed data into OHE. Unknown/unrecognized classes vector of all 0s.

        :param column_data: Pre-processed data to encode
        :returns: Encoded data of form :math:`N_{rows} x N_{categories}`
        """ # noqa
        if not self.is_prepared:
            raise Exception(
                'You need to call "prepare" before calling "encode" or "decode".'
            )

        ret = torch.zeros(size=(len(column_data), self.output_size))

        for idx, word in enumerate(column_data):
            index = self.map.get(word, None)

            if index is not None:
                ret[idx, index] = 1

            if self.use_unknown and index is None:
                ret[idx, 0] = 1

        return torch.Tensor(ret)

    def decode(self, encoded_data: torch.Tensor):
        """
        Decodes OHE mapping into the original categories. Since this approach uses an argmax, decoding flexibly works either on logits or an explicitly OHE vector.

        :param: encoded_data:
        :returns Returns the original category names for encoded data.
        """ # noqa
        encoded_data_list = encoded_data.tolist()
        ret = []

        for vector in encoded_data_list:

            all_zeros = not np.any(vector)
            if not self.use_unknown and all_zeros:
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
        """
        total = sum(vec)
        return [i / total for i in vec]
