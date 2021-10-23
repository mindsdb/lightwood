from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.api.types import PredictionArguments
from lightwood.data.encoded_ds import EncodedDs
from lightwood.ensemble import BaseEnsemble

import shap


class ShapleyValues(BaseAnalysisBlock):
    def __init__(self, deps: Optional[Tuple] = ...):
        super().__init__(deps=deps)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        predictor: BaseEnsemble = ns.predictor
        train_data: EncodedDs = ns.train_data

        def model(x: np.ndarray) -> np.ndarray:
            assert(isinstance(x, np.ndarray))
            # TODO(Andrea): is there a better way to call a model than to recreate its
            # EncodedDs every time?
            df = pd.DataFrame(data=x, columns=train_data.data_frame.columns)
            ds = EncodedDs(encoders=train_data.encoders, data_frame=df, target=train_data.target)

            # TODO(Andrea): why does this return an object when is should be an int??
            # TODO(Andrea): should I use dtype_dict to fix this?
            prediction = predictor(ds=ds, args=PredictionArguments()).astype(int).values

            return prediction

        # TODO(Andrea): I must exclude the target I suppose?
        explainer = shap.KernelExplainer(model=model, data=train_data.data_frame)

        info['shap_explainer'] = explainer

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object],
                **kwargs
                ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        ns = SimpleNamespace(**kwargs)

        shap_explainer = ns.analysis['shap_explainer']

        # TODO(Andrea): nsamples?
        shap_values = shap_explainer.shap_values(ns.data, silent=True)[0]
        shap_values_df = pd.DataFrame(shap_values).rename(
            mapper=lambda i: f"feature_{i}_impact", axis='columns')

        # TODO(Andrea): categorical??
        predictions = row_insights['prediction'].astype(int)

        base_response = (predictions - shap_values_df.sum(axis='columns')).mean()
        global_insights['base_response'] = base_response

        row_insights = pd.concat([row_insights, shap_values_df], axis='columns')

        breakpoint()

        return row_insights, global_insights
