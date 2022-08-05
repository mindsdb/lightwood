from copy import deepcopy
from types import SimpleNamespace
from typing import Dict

import numpy as np
from sklearn.utils import shuffle

from lightwood.helpers.log import log
from lightwood.data.encoded_ds import EncodedDs
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.api.types import PredictionArguments


class PermutationFeatureImportance(BaseAnalysisBlock):
    """
    Analysis block that estimates column importances via permutation.

    Roughly speaking, the procedure:
        - iterates over all input columns
        - if the input column is optional, shuffle its values, then generate predictions for the input data
        - compare this accuracy with the accuracy obtained using unshuffled data
        - all accuracy differences are normalized with respect to the original accuracy (clipped at zero if negative)
        - report these as estimated column importance scores

    Note that, crucially, this method does not refit the predictor at any point.

    :param row_limit: Set to 0 to use the entire validation dataset.
    :param col_limit: Set to 0 to consider all possible columns.

    Reference:
        https://scikit-learn.org/stable/modules/permutation_importance.html
        https://compstat-lmu.github.io/iml_methods_limitations/pfi.html
    """
    def __init__(self, disable_column_importance=False, row_limit=1000, col_limit=10, deps=tuple('AccStats',)):
        super().__init__(deps=deps)
        self.disable_column_importance = disable_column_importance
        self.row_limit = row_limit
        self.col_limit = col_limit
        self.n_decimals = 3

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if self.disable_column_importance:
            info['column_importances'] = None
        elif ns.tss.is_timeseries or ns.has_pretrained_text_enc:
            log.warning(f"Block 'PermutationFeatureImportance' does not support time series nor text encoding, skipping...")  # noqa
            info['column_importances'] = None
        else:
            if self.row_limit:
                log.info(f"[PFI] Using a random sample ({self.row_limit} rows out of {len(ns.encoded_val_data.data_frame)}).")  # noqa
                ref_df = ns.encoded_val_data.data_frame.sample(frac=1).reset_index(drop=True).iloc[:self.row_limit]
            else:
                log.info(f"[PFI] Using complete validation set ({len(ns.encoded_val_data.data_frame)} rows).")
                ref_df = deepcopy(ns.encoded_val_data.data_frame)

            ref_data = EncodedDs(ns.encoded_val_data.encoders, ref_df, ns.target)

            args = {'predict_proba': True} if ns.is_classification else {}
            ref_preds = ns.predictor(ref_data, args=PredictionArguments.from_dict(args))
            ref_score = np.mean(list(evaluate_accuracy(ref_data.data_frame,
                                                       ref_preds['prediction'],
                                                       ns.target,
                                                       ns.accuracy_functions
                                                       ).values()))
            shuffled_col_accuracy = {}
            shuffled_cols = []
            for x in ns.input_cols:
                if ('__mdb' not in x) and \
                        (not ns.tss.is_timeseries or (x != ns.tss.order_by and x not in ns.tss.historical_columns)):
                    shuffled_cols.append(x)

            if self.col_limit:
                shuffled_cols = shuffled_cols[:min(self.col_limit, len(ns.encoded_val_data.data_frame.columns))]
                log.info(f"[PFI] Set to consider first {self.col_limit} columns out of {len(shuffled_cols)}: {shuffled_cols}.")  # noqa
            else:
                log.info(f"[PFI] Computing importance for all {len(shuffled_cols)} columns: {shuffled_cols}")

            for col in shuffled_cols:
                shuffle_data = deepcopy(ref_data)
                shuffle_data.clear_cache()
                shuffle_data.data_frame[col] = shuffle(shuffle_data.data_frame[col].values)

                shuffled_preds = ns.predictor(shuffle_data, args=PredictionArguments.from_dict(args))
                shuffled_col_accuracy[col] = np.mean(list(evaluate_accuracy(
                    shuffle_data.data_frame,
                    shuffled_preds['prediction'],
                    ns.target,
                    ns.accuracy_functions
                ).values()))

            column_importances = {}
            acc_increases = np.zeros((len(shuffled_cols),))
            for i, col in enumerate(shuffled_cols):
                accuracy_increase = (ref_score - shuffled_col_accuracy[col])
                acc_increases[i] = round(accuracy_increase, self.n_decimals)
            for col, inc in zip(shuffled_cols, acc_increases):
                column_importances[col] = inc

            info['column_importances'] = column_importances

        return info
