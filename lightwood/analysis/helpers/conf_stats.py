from copy import deepcopy
from typing import Dict
from types import SimpleNamespace

from sklearn.preprocessing import OrdinalEncoder

from lightwood.analysis.base import BaseAnalysisBlock


class ConfStats(BaseAnalysisBlock):
    """
    Computes confidence-related statistics on the held-out validation dataset.

    TODO: regression & forecasting tasks
    """

    def __init__(self, deps=('ICP',), ece_bins: int = 10):
        super().__init__(deps=deps)
        self.ece_bins = ece_bins
        self.ordenc = OrdinalEncoder()
        self.n_decimals = 3

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if ns.is_classification:
            possible_labels = ns.stats_info.train_observed_classes
            self.ordenc.fit([[label] for label in possible_labels])
            task_type = 'categorical'
        elif ns.is_multi_ts:
            task_type = 'multi_ts'
        elif ns.is_numerical:
            task_type = 'numerical'
        else:
            return info

        ces, ece, mce, gscore = self._get_stats(info['result_df'],
                                                ns.normal_predictions,
                                                ns.data,
                                                ns.target,
                                                task_type)
        info['maximum_calibration_error'] = round(mce, self.n_decimals)
        info['expected_calibration_error'] = round(ece, self.n_decimals)
        info['binned_conf_acc_difference'] = ces
        info['global_calibration_score'] = round(gscore, self.n_decimals)
        return info

    def _get_stats(self, confs, preds, data, target, task_type='categorical'):
        """
        Computes expected and maximum calibration error for classification tasks.

        Amount of bins is specified by `self.ece_bins`. Data is sorted by increasing confidence prior to binning.

        :return:
        bins: bin-wise absolute difference between estimated confidence and true accuracy.
        ece: weighted average of all bins.
        mce: maximum value in `bins`.
        global_score: 1.0 minus absolute difference between accuracy and confidence over the entire validation set.
        """
        confs = deepcopy(confs).reset_index(drop=True)
        sorted_preds = deepcopy(preds).reset_index(drop=True)
        sorted_inp = deepcopy(data).reset_index(drop=True)
        sorted_val = confs.sort_values(by='confidence', kind='stable')
        sorted_inp['__mdb_confidence'] = sorted_val['confidence']

        if task_type == 'categorical':
            sorted_inp['__mdb_prediction'] = sorted_preds['prediction']
        else:
            if isinstance(confs['lower'][0], list):
                sorted_inp['__mdb_lower'] = confs['lower'].apply(lambda x: x[0])
                sorted_inp['__mdb_upper'] = confs['upper'].apply(lambda x: x[0])
            else:
                sorted_inp['__mdb_lower'] = confs['lower']
                sorted_inp['__mdb_upper'] = confs['upper']

            sorted_inp['__mdb_hits'] = (sorted_inp['__mdb_lower'] <= sorted_inp[target]) & \
                                       (sorted_inp[target] <= sorted_inp['__mdb_upper'])

        size = round(len(sorted_inp) / self.ece_bins)
        bins = []
        ece = 0

        for i in range(1, self.ece_bins):
            bin = sorted_inp.iloc[(i - 1) * size:i * size]

            if len(bin) > 0:
                if task_type == 'categorical':
                    acc = sum(bin[target] == bin['__mdb_prediction']) / size
                else:
                    acc = sum(bin['__mdb_hits'].astype(int)) / len(bin)

                conf = sum(bin['__mdb_confidence']) / size
                gap = abs(acc - conf)
                bins.append(gap)
                ece += gap

        ece /= self.ece_bins
        mce = max(bins) if bins else 0

        if task_type == 'categorical':
            global_acc = sum(sorted_inp[target] == sorted_inp['__mdb_prediction']) / len(sorted_inp)
        else:
            global_acc = sum(sorted_inp['__mdb_hits'].astype(int)) / len(sorted_inp)
        global_conf = sorted_inp['__mdb_confidence'].mean()
        global_score = 1.0 - abs(global_acc - global_conf)

        return bins, ece, mce, global_score
