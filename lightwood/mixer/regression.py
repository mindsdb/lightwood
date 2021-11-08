import torch
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LinearRegression

from lightwood.helpers.log import log
from lightwood.api.dtype import dtype
from lightwood.mixer import BaseMixer
from lightwood.encoder.base import BaseEncoder
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import ConcatedEncodedDs, EncodedDs
from lightwood.api.types import seconds


class Regression(BaseMixer):
    """
    A mixer that runs a simple linear regression using the (Encoded) features to predict the target.
    Supports all types because they are all encoded numerically.
    """ # noqa
    model: LinearRegression
    label_map: dict
    supports_proba: bool

    def __init__(self, stop_after: seconds, target_encoder: BaseEncoder, dtype_dict: dict, target: str):
        """
        :param stop_after: Maximum amount of time it should train for, currently ignored
        :param target_encoder: The encoder which will be used to decode the target
        :param dtype_dict: Data type dictionary
        :param target: Name of the target column
        """ # noqa
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        self.target_dtype = dtype_dict[target]
        self.supports_proba = self.target_dtype in [dtype.binary, dtype.categorical]
        self.label_map = {}
        self.stable = False

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the linear regression on the data, making it ready to predit

        :param train_data: The EncodedDs on which to fit the regression
        :param dev_data: Data used for early stopping and hyperparameter determination
        """
        if self.target_dtype not in (dtype.float, dtype.integer, dtype.quantity):
            raise Exception(f'Unspported {self.target_dtype} type for regression')
        log.info('Fitting Linear Regression model')
        X = []
        Y = []
        for x, y in ConcatedEncodedDs([train_data, dev_data]):
            X.append(x.tolist())
            Y.append(y.tolist())

        if self.supports_proba:
            self.label_map = self.target_encoder.rev_map

        self.model = LinearRegression().fit(X, Y)
        log.info(f'Regression based correlation of: {self.model.score(X, Y)}')

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """
        Fits the linear regression on some data, this refits the model entirely rather than updating it

        :param train_data: The EncodedDs on which to fit the regression
        :param dev_data: Data used for early stopping and hyperparameter determination
        """
        self.fit(train_data, dev_data)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        """
        Make predictions based on datasource similar to the one used to fit (sans the target column)

        :param ds: The EncodedDs for which to generate the predictions
        :param arg: Argument for predicting

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
