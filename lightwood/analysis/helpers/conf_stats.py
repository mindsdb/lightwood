from types import SimpleNamespace
from typing import Dict  # , Tuple

# import numpy as np
# import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from lightwood.helpers.log import log
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
        log.info(f"ECE - {round(ece, 4)}")
        log.info(f"MCE - {round(mce, 4)}")
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
        sorted_inp['__mdb_prediction'] = sorted_preds['prediction']
        mce = 0
        ces = []
        partial_ece = 0
        for i in range(1, self.ece_bins):
            size = round(len(sorted_inp) / self.ece_bins)
            bin = sorted_inp.iloc[size * (i - 1):i * size]
            acc = (1 / size) * sum(bin[target] == bin['__mdb_prediction'])
            conf = (1 / size) * sum(bin['__mdb_confidence'])
            ce = (abs(acc - conf))
            ces.append(ce)
            partial_ece += ce * size
            mce = abs(acc - conf) if abs(acc - conf) > mce else mce
        ece = partial_ece / self.ece_bins
        return ces, ece, mce
