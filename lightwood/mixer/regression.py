import torch
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import Ridge

from lightwood.helpers.log import log
from lightwood.api.dtype import dtype
from lightwood.mixer import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs


class Regression(BaseMixer):
    """
    The `Regression` mixer inherits from scikit-learn's `Ridge` class
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
    
    This class performs Ordinary Least-squares Regression (OLS) under the hood; 
    this means it fits a set of coefficients (w_1, w_2, ..., w_N) for an N-length feature vector, that minimize the difference
    between the predicted target value and the observed true value.
  
    This mixer intakes featurized (encoded) data to predict the target. It deploys if the target data-type is considered numerical/integer.
    """ # noqa
    model: Ridge
    label_map: dict
    supports_proba: bool

    def __init__(self, stop_after: float, target_encoder: BaseEncoder, dtype_dict: dict, target: str):
        """
        :param stop_after: Maximum amount of seconds it should fit for, currently ignored
        :param target_encoder: The encoder which will be used to decode the target
        :param dtype_dict: A map of feature names and their data types
        :param target: Name of the target column
        """ # noqa
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.target_dtype = dtype_dict[target]
        self.dtype_dict = dtype_dict
        self.supports_proba = self.target_dtype in [dtype.binary, dtype.categorical]
        self.label_map = {}
        self.stable = False

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits `Ridge` model on input feature data to provide predictions.

        :param train_data: Regression if fit on this
        :param dev_data: This just gets concatenated to the ``train_data``
        """
        if self.target_dtype not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(f'Unspported {self.target_dtype} type for regression')

        if self.stop_after < len(train_data) * len(self.dtype_dict) / pow(10, 3):
            raise Exception(f'Insufficient time ({self.stop_after} seconds) to fit a linear regression on the data!')

        log.info('Fitting Linear Regression model')
        X = []
        Y = []
        for x, y in ConcatedEncodedDs([train_data, dev_data]):
            X.append(x.tolist())
            Y.append(y.tolist())

        if self.supports_proba:
            self.label_map = self.target_encoder.rev_map

        self.model = Ridge().fit(X, Y)
        log.info(f'Regression based correlation of: {self.model.score(X, Y)}')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the linear regression on some data, this refits the model entirely rather than updating it

        :param train_data: Regression is fit on this
        :param dev_data: This just gets concatenated to the ``train_data``
        """
        self.fit(train_data, dev_data)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Make predictions based on datasource with the same features as the ones used for fitting

        :param ds: Predictions are generate from it
        :param arg: Any additional arguments used in predicting

        :returns: A dataframe cotaining the decoded predictions and (depending on the args) additional information such as the probabilites for each target class
        """ # noqa
        X = []
        for x, _ in ds:
            X.append(x.tolist())

        Yh = self.model.predict(X)

        decoded_predictions = self.target_encoder.decode(torch.Tensor(Yh))

        ydf = pd.DataFrame({'prediction': decoded_predictions})

        if args.predict_proba and self.label_map:
            raw_predictions = softmax(Yh.squeeze(), axis=1)
            for idx, label in enumerate(self.target_encoder.rev_map.values()):
                ydf[f'__mdb_proba_{label}'] = raw_predictions[:, idx]

        return ydf
