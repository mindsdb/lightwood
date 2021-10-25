import warnings
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.api import dtype
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.ensemble import BaseEnsemble
from lightwood.helpers.log import log
from sklearn.preprocessing import LabelEncoder

import shap


class ShapleyValues(BaseAnalysisBlock):
    label_encoder: LabelEncoder

    def __init__(self, deps: Optional[Tuple] = ...):
        super().__init__(deps=deps)
        self.label_encoder = LabelEncoder()

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        log.info('ShapleyValues analyze')
        ns = SimpleNamespace(**kwargs)

        output_dtype = ns.dtype_dict[ns.target]
        train_data: EncodedDs = ns.train_data

        if output_dtype in (dtype.integer, dtype.float, dtype.quantity):
            pass
        elif output_dtype in (dtype.binary, dtype.categorical, dtype.tags):
            self.label_encoder.fit(train_data.data_frame[ns.target].values)
        else:
            log.error(f'ShapleyValues analyzers not supported for type: {output_dtype}')
            raise Exception(f'ShapleyValues analyzers not supported for type: {output_dtype}')

        predictor: BaseEnsemble = ns.predictor

        def model(x: np.ndarray) -> np.ndarray:
            assert(isinstance(x, np.ndarray))
            df = pd.DataFrame(data=x, columns=train_data.data_frame.columns)
            ds = EncodedDs(encoders=train_data.encoders, data_frame=df, target=train_data.target)

            decoded_predictions = predictor(ds=ds, args=PredictionArguments())
            encoded_predictions = self.label_encoder.transform(decoded_predictions['prediction'].values)

            return encoded_predictions

        explainer = shap.KernelExplainer(model=model, data=train_data.data_frame)

        info['shap_explainer'] = explainer

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object],
                **kwargs
                ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        log.info('ShapleyValues explain')
        ns = SimpleNamespace(**kwargs)

        shap_explainer = ns.analysis['shap_explainer']

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            shap_values = shap_explainer.shap_values(ns.data, silent=True)

        shap_values_df = pd.DataFrame(shap_values).rename(
            mapper=lambda i: f"feature_{i}_impact", axis='columns')

        predictions = self.label_encoder.transform(row_insights['prediction'])

        base_response = (predictions - shap_values_df.sum(axis='columns')).mean()
        global_insights['base_response'] = base_response

        row_insights = pd.concat([row_insights, shap_values_df], axis='columns')

        return row_insights, global_insights
