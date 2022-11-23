import warnings
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightwood.analysis.base import BaseAnalysisBlock
from type_infer.dtype import dtype
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.helpers.log import log
from sklearn.preprocessing import LabelEncoder

import shap


class ShapleyValues(BaseAnalysisBlock):
    """
    Analysis block that estimates column importance with SHAP (SHapley Additive exPlanations), a game theoretic approach
    to explain the ouput of any machine learning model. SHAP assigns each feature an importance value for a particular
    prediction.

    Reference:
        https://shap.readthedocs.io/en/stable/
        https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
    """
    label_encoder: LabelEncoder

    def __init__(self, deps: Optional[Tuple] = ...):
        super().__init__(deps=deps)
        self.label_encoder = LabelEncoder()
        self.columns = []
        self.target = None

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        log.info('Preparing to compute feature importance values with SHAP')
        ns = SimpleNamespace(**kwargs)

        output_dtype = ns.dtype_dict[ns.target]
        train_data: EncodedDs = ns.train_data

        if output_dtype in (dtype.integer, dtype.float, dtype.quantity):
            pass
        elif output_dtype in (dtype.binary, dtype.categorical, dtype.tags):
            self.label_encoder.fit(train_data.data_frame[ns.target].values)
        else:
            log.warning(f'ShapleyValues analyzers not supported for type: {output_dtype}')
            return info

        self.target = ns.target
        self.columns = list(set(ns.dtype_dict.keys()) - {self.target})
        input_df = train_data.data_frame[self.columns]

        def model(x: np.ndarray) -> np.ndarray:
            assert(isinstance(x, np.ndarray))
            df = pd.DataFrame(data=x, columns=self.columns)
            ds = EncodedDs(encoders=train_data.encoders, data_frame=df, target=train_data.target)

            decoded_predictions = ns.predictor(ds=ds, args=PredictionArguments())
            if output_dtype in (dtype.integer, dtype.float, dtype.quantity):
                encoded_predictions = decoded_predictions['prediction'].values
            elif output_dtype in (dtype.binary, dtype.categorical, dtype.tags):
                encoded_predictions = self.label_encoder.transform(decoded_predictions['prediction'].values)

            return encoded_predictions

        info['shap_explainer'] = shap.KernelExplainer(model=model, data=input_df)

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object],
                **kwargs
                ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        log.info('Computing feature importance values with Kernel SHAP method')
        ns = SimpleNamespace(**kwargs)

        shap_explainer = ns.analysis.get('shap_explainer', None)
        if shap_explainer is None:
            return row_insights, global_insights

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            shap_values = shap_explainer.shap_values(ns.data[self.columns], silent=True)

        shap_values_df = pd.DataFrame(shap_values).rename(
            mapper=lambda i: f"shap_contribution_{self.columns[i]}", axis='columns')

        if kwargs.get('target_dtype', None) in (dtype.binary, dtype.categorical, dtype.tags):
            predictions = self.label_encoder.transform(row_insights['prediction'])
        else:
            predictions = row_insights['prediction']

        base_response = (predictions - shap_values_df.sum(axis='columns')).mean()

        row_insights = pd.concat([row_insights, shap_values_df], axis='columns')
        row_insights['shap_base_response'] = base_response
        row_insights['shap_final_response'] = predictions

        return row_insights, global_insights
