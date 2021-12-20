from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from lightwood.api.dtype import dtype
from lightwood.analysis.base import BaseAnalysisBlock


class ConfStats(BaseAnalysisBlock):
    """ Computes confidence-related statistics on the held-out validation dataset. """

    def __init__(self, deps=('ICP',), ece_bins: int = 10):
        super().__init__(deps=deps)
        self.ece_bins = ece_bins

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)
        print(f"ECE is: {self.ece(ns)}")

        return info

    def ece(self, ns):
        """ Computes expected calibration error. """
        sorted_val = ns.normal_predictions.sort_values(by='confidence')
        sorted_inp = ns.data.reindex(sorted_val.index)
        sorted_inp['__mdb_confidence'] = sorted_val['confidence']
        pairs = []
        for i in range(1, self.ece_bins):
            interval = sorted_inp.iloc[len(sorted_inp)*(i-1)/self.ece_bins:i*len(sorted_inp)/self.ece_bins]
            pos = (1/len(interval))*sum(sorted_inp[ns.target])
            prb = (1/len(interval))*sum(sorted_inp['__mdb_confidence'])
            pairs.append((pos, prb))

        total_ece = 0
        for i in range(1, self.ece_bins):
            interval = sorted_inp.iloc[len(sorted_inp) * (i - 1) / self.ece_bins:i * len(sorted_inp) / self.ece_bins]
            ece = (len(interval)/len(sorted_inp))*((pairs[i-1][1] - pairs[i-1][0])**2)
            total_ece += ece

        return np.sqrt(total_ece)

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        return row_insights, global_insights

