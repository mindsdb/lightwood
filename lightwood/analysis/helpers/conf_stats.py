from types import SimpleNamespace
from typing import Dict, Tuple

# import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# from lightwood.api.dtype import dtype
from lightwood.analysis.base import BaseAnalysisBlock


class ConfStats(BaseAnalysisBlock):
    """ Computes confidence-related statistics on the held-out validation dataset. """

    def __init__(self, deps=('ICP',), ece_bins: int = 10):
        super().__init__(deps=deps)
        self.ece_bins = ece_bins
        self.ordenc = OrdinalEncoder()

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)
        # if ns.categorical # TODO regression
        possible_labels = ns.stats_info.train_observed_classes
        self.ordenc.fit([[label] for label in possible_labels])
        ces, ece, mce = self.ce(info['result_df'], ns.normal_predictions, ns.data, ns.target)
        print(f"CEs are: {ces}")
        print(f"ECE is: {ece}")
        print(f"MCE is: {mce}")
        info['CE'] = ces
        info['ECE'] = ece
        info['MCE'] = mce

        return info

    def ce(self, confs, preds, data, target):
        """ Computes expected and maximum calibration error. """
        sorted_val = confs.sort_values(by='confidence', kind='stable')
        sorted_preds = preds.reindex(sorted_val.index)
        sorted_inp = data.reindex(sorted_val.index)
        sorted_inp['__mdb_confidence'] = sorted_val['confidence']
        mce = 0
        ces = []
        partial_ece = 0
        for i in range(1, self.ece_bins):
            interval = sorted_inp.iloc[int(len(sorted_inp) * (i - 1) / self.ece_bins):int(i * len(sorted_inp) / self.ece_bins)]  # noqa
            acc = (1 / len(interval)) * sum(sorted_inp[target] == sorted_preds['prediction'])
            conf = (1 / len(interval)) * sum(sorted_inp['__mdb_confidence'])
            ce = (abs(acc - conf))
            ces.append(ce)
            partial_ece += ce * len(interval)
            mce = abs(acc - conf) if abs(acc - conf) > mce else mce
        ece = partial_ece / self.ece_bins
        return ces, ece, mce
