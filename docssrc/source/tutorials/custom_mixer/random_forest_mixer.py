
from lightwood.mixer import BaseMixer
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs
from lightwood import dtype
from lightwood.encoder import BaseEncoder

import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RandomForestMixer(BaseMixer):
    clf: RandomForestClassifier

    def __init__(self, stop_after: int, dtype_dict: dict, target: str, target_encoder: BaseEncoder):
        super().__init__(stop_after)
        self.target_encoder = target_encoder
        # Throw in case someone tries to use this for a problem that's not classification, I'd fail anyway, but this way the error message is more intuitive
        if dtype_dict[target] not in (dtype.categorical, dtype.binary):
            raise Exception(f'This mixer can only be used for classification problems! Got target dtype {dtype_dict[target]} instead!')

        # We could also initialize this in `fit` if some of the parameters depend on the input data, since `fit` is called exactly once
        self.clf = RandomForestClassifier(max_depth=30)

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        X, Y = [], []
        # By default mixers get some train data and a bit of dev data on which to do early stopping or hyper parameter optimization. For this mixer, we don't need dev data, so we're going to concat the two in order to get more training data. Then, we're going to turn them into an sklearn friendly foramat.
        for x, y in ConcatedEncodedDs([train_data, dev_data]):
            X.append(x.tolist())
            Y.append(y.tolist())
        self.clf.fit(X, Y)

    def __call__(self, ds: EncodedDs,
                 args: PredictionArguments = PredictionArguments()) -> pd.DataFrame:
        # Turn the data into an sklearn friendly format
        X = []
        for x, _ in ds:
            X.append(x.tolist())

        Yh = self.clf.predict(X)

        # Lightwood encoders are meant to decode torch tensors, so we have to cast the predictions first
        decoded_predictions = self.target_encoder.decode(torch.Tensor(Yh))

        # Finally, turn the decoded predictions into a dataframe with a single column called `prediction`. This is the standard behaviour all lightwood mixers use
        ydf = pd.DataFrame({'prediction': decoded_predictions})

        return ydf

    
    # We'll skip implementing `partial_fit`, thus making this mixer unsuitable for online training tasks
