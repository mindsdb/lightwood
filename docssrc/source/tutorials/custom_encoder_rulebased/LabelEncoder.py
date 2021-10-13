"""
2021.10.13

Create a LabelEncoder that transforms categorical data into a label.
"""
import pandas as pd
import torch

from lightwood.encoder import BaseEncoder

class LabelEncoder:
    """
    Create a label representation for categorical data. The data will rely on sorted to organize the order of the labels.

    Class Attributes:
    - is_target: Whether this is used to encode the target
    - is_prepared: Whether the encoder rules have been set (after ``prepare`` is called)
        
    """ # noqa
    is_target: bool
    is_prepared: bool

    is_timeseries_encoder: bool = False
    is_trainable_encoder: bool = False

    def __init__(self, is_target: bool =False) -> None:
        """
        Initialize the Label Encoder

        :param is_target: 
        """
        self.is_target = is_target
        self.is_prepared = False
        self.output_size = None

    # Not all encoders need to be prepared
    def prepare(self, priming_data) -> None:
        """
        Create a LabelEncoder for categorical data.

        LabelDict creates a mapping where each index is associated to a category.

        :param priming_data: Input column data that is categorical.

        :returns: Nothing; prepares encoder rules with `label_dict` and `ilabel_dict`
        """

        # Find all unique categories in the dataset
        categories = priming_data.unique()

        log.info("Categories Detected = " + str(len(categories)))

        self.label_dict = {"Unknown": 0} # Include an unknown category
        self.label_dict.update({cat: idx+1 for idx, cat in enumerate(categories)})
        self.ilabel_dict = {idx: cat for cat, idx in self.label_dict.items()} 

        self.is_prepared = True

    def encode(self, column_data: pd.Series) -> torch.Tensor:
        """
        Convert pre-processed data into the labeled values

        :param column_data: Pandas series to convert into labels
        """
        enc = column_data.apply(lambda x: self.label_dict.get(x, 0))
        return torch.Tensor(enc.tolist()).int()

    def decode(self, encoded_data: torch.Tensor) -> List[object]:
        """
        Convert torch.Tensor labels into categorical data
        """
        return [self.ilabel_dict[i.item()] for i in encoded_data]