from types import SimpleNamespace
from typing import Dict, Tuple

# import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from lightwood.analysis.nn_conf.platt import PlattCalibrator
# from lightwood.api.dtype import dtype
from lightwood.analysis.base import BaseAnalysisBlock


class NNRegressionCalibrator(BaseAnalysisBlock):
    """ Trains an NN that learns to output a confidence score and bounds for numerical targets.
    TODO: rm these notes later
    numerical:  either
    1) jorge's idea:
        train:
            NN(features, prediction) -> predicts (normal) distribution parameters that when sampled gives the residual

        predict(n_samples=100):
            NN(features, prediction) -> get params, then sample `n_samples` errors from it to get a prediction distribution
                you get a min/max bound.
    2) quantile regression over the residuals!

    """  # noqa

    def __init__(self):
        super().__init__()

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        self.fit(ns, info['result_df'])
        return info

    def fit(self, ns: SimpleNamespace):
        pass

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        return row_insights


class NNClassificationCalibrator(BaseAnalysisBlock):
    """ Platt scaling to output a calibrated confidence score for categorical targets.
    TODO: rm these notes later
    categorical:
        train: fit a platt scaling
        test: you get confidence scores (i.e. how many votes does the majority class have)
    """

    def __init__(self):
        super().__init__()
        self.calibrator = PlattCalibrator()
        self.ordenc = OrdinalEncoder()

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)
        possible_labels = ns.stats_info.train_observed_classes
        self.ordenc.fit([[label] for label in possible_labels])

        true = ns.data[ns.target].values.reshape(-1, 1).astype(float)
        pred = ns.normal_predictions['prediction'].values.reshape(-1, 1).astype(float)
        self.calibrator.fit(pred, true)
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        row_insights['confidence'] = self.calibrator.predict(self.ordenc.transform(
            row_insights['prediction'].values.reshape(-1, 1)))
        return row_insights, global_insights
