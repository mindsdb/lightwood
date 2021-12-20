from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from lightwood.api.dtype import dtype
from lightwood.mixer import BaseMixer
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class Normalizer(BaseMixer):
    """
    Companion class to the confidence estimation analysis block. A normalizer is a secondary machine learning model
    tasked with learning to estimate the "difficulty" that the main predictor will have with any problem instance.

    The idea is that this model should emit higher scores for tougher predictions. All scores will be passed as a
    normalizing factor to the conformal prediction framework, thus:
      - widening bounds at the same confidence level if a prediction is harder
      - tightening bounds at the same confidence level if a predictions is easier
      
    Reference:
        Papadopoulos, H., Gammerman, A., & Vovk, V. (2008). Normalized nonconformity measures for regression Conformal Prediction.

    """  # noqa
    def __init__(self, fit_params: dict):
        super(Normalizer, self).__init__(stop_after=fit_params['stop_after'])

        self.input_cols = list(fit_params['dtype_dict'].keys())
        self.base_predictor = fit_params['predictor']
        self.encoders = fit_params['encoders']
        self.target = fit_params['target']
        self.target_dtype = fit_params['dtype_dict'][fit_params['target']]
        self.multi_ts_task = fit_params['is_multi_ts']

        self.model = Ridge()  # TODO: enable underlying model selection from JsonAI
        self.prepared = False
        self.prediction_cache = None
        self.bounds = (0.5, 1.5)
        self.error_fn = mean_absolute_error

    def fit(self, data: EncodedDs) -> None:
        try:
            data = ConcatedEncodedDs([data])
            preds = self.base_predictor(data, args=PredictionArguments.from_dict({'predict_proba': True}))
            truths = data.data_frame[self.target]
            labels = self.get_labels(preds, truths.values, data.encoders[self.target])
            enc_data = data.get_encoded_data(include_target=False).numpy()
            self.model.fit(enc_data, labels)
            self.prepared = True
        except Exception:
            pass

    def __call__(self, ds: Union[ConcatedEncodedDs, EncodedDs, np.ndarray], args: PredictionArguments) \
            -> np.ndarray:
        if isinstance(ds, EncodedDs) or isinstance(ds, ConcatedEncodedDs):
            ds = ds.get_encoded_data(include_target=False).numpy()  # TODO: set upper limit for speed

        if self.prepared:
            raw = self.model.predict(ds)
            scores = np.clip(raw, 0.1, 1e4)  # set limit deviations (TODO: benchmark stability)
            # scores = scores / scores.mean()  # smoothed
        else:
            scores = np.ones(ds.shape[0])

        return scores

    def score(self, data) -> np.ndarray:
        if not self.prepared:
            scores = np.ones(data.shape[0])  # by default, normalizing factor is 1 for all predictions
        elif self.prediction_cache is not None:
            scores = self.prediction_cache
        else:
            scores = self.model.predict(data)

        return scores

    def get_labels(self, preds: pd.DataFrame, truths: np.ndarray, target_enc) -> np.ndarray:
        if self.target_dtype in [dtype.integer, dtype.float, dtype.quantity]:
            if not self.multi_ts_task:
                preds = preds.values.squeeze()
            else:
                preds = pd.Series([p[0] for p in preds.values.squeeze()]).values
            preds = preds.astype(float)
            labels = self.compute_numerical_labels(preds, truths, self.bounds)

        elif self.target_dtype in [dtype.binary, dtype.categorical]:
            if self.base_predictor.supports_proba:
                prob_cols = [col for col in preds.columns if '__mdb_proba' in col]
                col_names = [col.replace('__mdb_proba_', '') for col in prob_cols]
                preds = preds[prob_cols]
            else:
                prob_cols = col_names = target_enc.map.keys()
                ohe_preds = pd.get_dummies(preds['prediction'], columns=col_names)
                for col in col_names:
                    if col not in ohe_preds.columns:
                        ohe_preds[col] = np.zeros(ohe_preds.shape[0])
                preds = ohe_preds

            # reorder preds to ensure classes are in same order as in target_enc
            preds.columns = col_names
            new_order = [v for k, v in sorted(target_enc.rev_map.items(), key=lambda x: x[0])]
            preds = preds.reindex(columns=new_order)
            preds = preds.values.squeeze()
            preds = preds if prob_cols else target_enc.encode(preds).tolist()
            truths = target_enc.encode(truths).numpy()

            labels = self.compute_categorical_labels(preds, truths)

        else:
            raise(Exception(f"dtype {self.target_dtype} not supported for confidence normalizer"))

        return labels

    @staticmethod
    def compute_numerical_labels(preds: np.ndarray, truths: np.ndarray, bounds: list) -> np.ndarray:
        diffs = np.log(abs(preds - truths))
        diffs = diffs / np.max(diffs) if np.max(diffs) > 0 else diffs
        labels = np.clip(bounds[0] + diffs, bounds[0], bounds[1])
        return labels

    @staticmethod
    def compute_categorical_labels(preds: np.ndarray, truths: np.ndarray) -> np.ndarray:
        preds = np.clip(preds, 0.001, 0.999)  # avoid inf
        labels = entropy(truths, preds, axis=1)
        return labels
