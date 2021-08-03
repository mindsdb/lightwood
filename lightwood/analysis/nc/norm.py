from typing import Union, List

import torch
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from lightwood.api.dtype import dtype
from lightwood.model import BaseModel
from lightwood.data.encoded_ds import EncodedDs, ConcatedEncodedDs


class Normalizer(BaseModel):
    def __init__(self, fit_params: dict):
        super(Normalizer, self).__init__(stop_after=fit_params['stop_after'])

        self.input_cols = list(fit_params['dtype_dict'].keys())
        self.base_predictor = fit_params['predictor']
        self.encoders = fit_params['encoders']
        self.target = fit_params['target']
        self.target_dtype = fit_params['dtype_dict'][fit_params['target']]
        self.multi_ts_task = fit_params['is_multi_ts']

        self.model = Ridge()
        self.prediction_cache = None
        self.bounds = (0.5, 1.5)
        self.error_fn = mean_absolute_error

    def fit(self, data: List[EncodedDs]) -> None:
        data = ConcatedEncodedDs(data)
        preds = self.base_predictor(data, predict_proba=True)
        truths = data.data_frame[self.target]
        labels = self.get_labels(preds, truths.values, data.encoders[self.target])
        enc_data = data.get_encoded_data(include_target=False).numpy()
        self.model.fit(enc_data, labels)

    def __call__(self, ds: Union[ConcatedEncodedDs, torch.Tensor], predict_proba: bool = False) -> np.ndarray:
        if isinstance(ds, ConcatedEncodedDs):
            ds = ds.get_encoded_data(include_target=False)
        raw = self.model.predict(ds.numpy())
        clipped = np.clip(raw, 0.1, 1e4)  # set limit deviations (@TODO: benchmark stability)
        # smoothed = clipped / clipped.mean()
        return clipped

    def score(self, data) -> np.ndarray:
        scores = self.prediction_cache if self.prediction_cache is not None else self.model.predict(data)

        if scores is None:
            scores = np.ones(data.shape[0])  # by default, normalizing factor is 1 for all predictions
        return scores

    def get_labels(self, preds: pd.DataFrame, truths: np.ndarray, target_enc) -> np.ndarray:
        if self.target_dtype in [dtype.integer, dtype.float]:

            if not self.multi_ts_task:
                preds = preds.values.squeeze()
            else:
                preds = [p[0] for p in preds.values.squeeze()]

            diffs = np.log(abs(preds - truths))
            labels = np.clip(self.bounds[0] + diffs / np.max(diffs), self.bounds[0], self.bounds[1])

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

            # get log loss
            preds = preds.values.squeeze()
            preds = preds if prob_cols else target_enc.encode(preds).tolist()
            preds = np.clip(preds, 0.001, 0.999)  # avoid inf
            truths = target_enc.encode(truths).numpy()
            labels = entropy(truths, preds, axis=1)

        else:
            raise(Exception(f"dtype {self.target_dtype} not supported for confidence normalizer"))

        return labels
