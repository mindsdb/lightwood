from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.suod import SUOD

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.log import log
from lightwood.helpers.parallelism import get_nr_procs


class PyOD(BaseAnalysisBlock):
    """
    Wrapper analysis block for the 'PyOD' anomaly detection library.

    For now, the following techniques are supported:
      - SUOD: Large-scale unsupervised heterogeneous outlier detection
    """  # noqa

    def __init__(self, contamination=0.1, deps: Optional[Tuple] = ...):
        super().__init__(deps=deps)
        self.target = None
        self.input_cols = []
        self.exceptions = ['__make_predictions']
        self.ordinal_encoders = dict()
        if contamination > 0.5:
            log.warning("Contamination higher than maximum possible value, setting at 0.5...")
        self.contamination = min(contamination, 0.5)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        log.info('Preparing to compute anomaly detection with PyOD')
        ns = SimpleNamespace(**kwargs)
        self.target = ns.target
        detector_list = [LOF(n_neighbors=10, contamination=self.contamination),
                         LOF(n_neighbors=30, contamination=self.contamination),
                         HBOS(contamination=self.contamination),
                         PCA(contamination=self.contamination)]

        df = deepcopy(ns.train_data.data_frame)
        n_jobs = get_nr_procs(df)
        clf = SUOD(base_estimators=detector_list,
                   contamination=self.contamination,
                   n_jobs=n_jobs,
                   combination='average',
                   verbose=False)

        if ns.tss.is_timeseries:
            for gcol in ns.tss.group_by:
                self.ordinal_encoders[gcol] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                self.ordinal_encoders[gcol].fit(df[gcol].values.reshape(-1, 1))
            for i in range(1, ns.tss.horizon):
                self.exceptions.append(f'{self.target}_timestep_{i}')

            self.exceptions.append(ns.tss.order_by)
            df = self._preprocess_ts_df(df, ns)

        for col in df.columns:
            if col != self.target and '__mdb' not in col and col not in self.exceptions:
                self.input_cols.append(col)

        df = df.loc[:, self.input_cols].dropna()
        clf.fit(df.values)
        info['pyod_explainer'] = clf
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object],
                **kwargs
                ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        log.info('Performing anomaly detection with PyOD...')
        ns = SimpleNamespace(**kwargs)
        pyod_explainer = ns.analysis.get('pyod_explainer', None)
        if pyod_explainer is None:
            return row_insights, global_insights

        df = deepcopy(ns.data)
        if ns.tss.is_timeseries:
            df = self._preprocess_ts_df(df, ns)

        df = df[self.input_cols].fillna(0)
        row_insights['pyod_anomaly'] = pyod_explainer.predict(df).astype(bool)  # binary labels (0: inlier, 1: outlier)
        return row_insights, global_insights

    def _preprocess_ts_df(self, df: pd.DataFrame, ns: SimpleNamespace) -> pd.DataFrame:
        for gcol in ns.tss.group_by:
            df[gcol] = self.ordinal_encoders[gcol].transform(df[gcol].values.reshape(-1, 1).astype(np.object)).flatten()

        for w in range(ns.tss.window + 1):
            df[f'__pyod_window_{ns.tss.window - w}'] = df[f'__mdb_ts_previous_{self.target}'].apply(lambda x: x[w])
        return df
