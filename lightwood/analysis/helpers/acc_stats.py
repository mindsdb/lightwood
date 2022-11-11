import random
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix

from type_infer.dtype import dtype
from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.general import evaluate_accuracy
from lightwood.helpers.log import log


class AccStats(BaseAnalysisBlock):
    """ Computes accuracy stats and a confusion matrix for the validation dataset """

    def __init__(self, deps=('ICP',)):
        super().__init__(deps=deps)
        self.n_decimals = 3

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        ns = SimpleNamespace(**kwargs)

        if ns.accuracy_functions == ['evaluate_array_accuracy'] and ns.ts_analysis.get('ts_naive_mae', {}):
            accuracy_functions = ['bounded_ts_accuracy']
            log.info("AccStats will bound the array accuracy for reporting purposes. Check `bounded_ts_accuracy` for a description of the bounding procedure.")  # noqa
        else:
            accuracy_functions = ns.accuracy_functions

        info['score_dict'] = evaluate_accuracy(ns.data, ns.normal_predictions['prediction'],
                                               ns.target, accuracy_functions, ts_analysis=ns.ts_analysis)

        info['normal_accuracy'] = round(np.mean(list(info['score_dict'].values())), self.n_decimals)
        self.fit(ns, info['result_df'])
        info['val_overall_acc'], info['acc_histogram'], info['cm'], info['acc_samples'] = self.get_accuracy_stats()
        return info

    def fit(self, ns: SimpleNamespace, conf=Optional[np.ndarray]):
        self.col_stats = ns.dtype_dict
        self.target = ns.target
        self.input_cols = ns.input_cols
        self.buckets = ns.stats_info.buckets if ns.stats_info.buckets else {}

        self.normal_predictions_bucketized = []
        self.real_values_bucketized = []
        self.numerical_samples_arr = []

        for n in range(len(ns.normal_predictions)):
            row = ns.data.iloc[n]
            real_value = row[self.target]
            predicted_value = ns.normal_predictions.iloc[n]['prediction']

            if isinstance(predicted_value, list):
                # T+N time series, for now we compare the T+1 prediction only @TODO: generalize
                predicted_value = predicted_value[0]

            predicted_value = predicted_value \
                if self.col_stats[self.target] not in [dtype.integer, dtype.float, dtype.quantity] \
                else float(predicted_value)

            real_value = real_value \
                if self.col_stats[self.target] not in [dtype.integer, dtype.float, dtype.quantity] \
                else float(real_value)

            if self.buckets:
                bucket = self.buckets[self.target]
                predicted_value_b = get_value_bucket(predicted_value, bucket, self.col_stats[self.target])
                real_value_b = get_value_bucket(real_value, bucket, self.col_stats[self.target])
            else:
                predicted_value_b = predicted_value
                real_value_b = real_value

            if conf is not None and self.col_stats[self.target] in [dtype.integer, dtype.float, dtype.quantity]:
                predicted_range = conf.iloc[n][['lower', 'upper']].tolist()
            else:
                predicted_range = (predicted_value_b, predicted_value_b)

            self.real_values_bucketized.append(real_value_b)
            self.normal_predictions_bucketized.append(predicted_value_b)
            if conf is not None and self.col_stats[self.target] in [dtype.integer, dtype.float, dtype.quantity]:
                self.numerical_samples_arr.append((real_value, predicted_range))

    def get_accuracy_stats(self, is_classification=None, is_numerical=None):
        bucket_accuracy = {}
        bucket_acc_counts = {}
        for i, bucket in enumerate(self.normal_predictions_bucketized):
            if bucket not in bucket_acc_counts:
                bucket_acc_counts[bucket] = []

            if len(self.numerical_samples_arr) != 0:
                bucket_acc_counts[bucket].append(self.numerical_samples_arr[i][1][0] <
                                                 self.numerical_samples_arr[i][0] < self.numerical_samples_arr[i][1][1]) # noqa
            else:
                bucket_acc_counts[bucket].append(1 if bucket == self.real_values_bucketized[i] else 0)

        for bucket in bucket_acc_counts:
            bucket_accuracy[bucket] = sum(bucket_acc_counts[bucket]) / len(bucket_acc_counts[bucket])

        accuracy_count = []
        for counts in list(bucket_acc_counts.values()):
            accuracy_count += counts

        overall_accuracy = round(sum(accuracy_count) / len(accuracy_count), self.n_decimals)

        for bucket in range(len(self.buckets)):
            if bucket not in bucket_accuracy:
                if bucket in self.real_values_bucketized:
                    # If it was never predicted, but it did exist as a real value, then assume 0% confidence when it does get predicted # noqa
                    bucket_accuracy[bucket] = 0

        for bucket in range(len(self.buckets)):
            if bucket not in bucket_accuracy:
                # If it wasn't seen either in the real values or in the predicted values, assume average confidence (maybe should be 0 instead ?) # noqa
                bucket_accuracy[bucket] = overall_accuracy

        accuracy_histogram = {
            'buckets': list(bucket_accuracy.keys()),
            'accuracies': list(bucket_accuracy.values()),
            'is_classification': is_classification,
            'is_numerical': is_numerical
        }

        labels = list(set([*self.real_values_bucketized, *self.normal_predictions_bucketized]))
        matrix = confusion_matrix(self.real_values_bucketized, self.normal_predictions_bucketized, labels=labels)
        matrix = [[int(y) if str(y) != 'nan' else 0 for y in x] for x in matrix]

        target_bucket = self.buckets[self.target]
        bucket_values = [target_bucket[i] if i < len(target_bucket) else None for i in labels]

        cm = {
            'matrix': matrix,
            'predicted': bucket_values,
            'real': bucket_values
        }

        accuracy_samples = None
        if len(self.numerical_samples_arr) > 0:
            nr_samples = min(400, len(self.numerical_samples_arr))
            sampled_numerical_samples_arr = random.sample(self.numerical_samples_arr, nr_samples)
            accuracy_samples = {
                'y': [x[0] for x in sampled_numerical_samples_arr],
                'x': [x[1] for x in sampled_numerical_samples_arr]
            }

        return overall_accuracy, accuracy_histogram, cm, accuracy_samples


def get_value_bucket(value, buckets, target_dtype):
    """
    :return: The bucket in the `histogram` in which our `value` falls
    """
    if buckets is None:
        return None

    if target_dtype in (dtype.binary, dtype.categorical):
        if value in buckets:
            bucket = buckets.index(value)
        else:
            bucket = len(buckets)  # for null values

    elif target_dtype in (dtype.integer, dtype.float, dtype.quantity):
        bucket = closest(buckets, value)
    else:
        bucket = len(buckets)  # for null values

    return bucket


def closest(arr, value):
    """
    :return: The index of the member of `arr` which is closest to `value`
    """
    if value is None:
        return -1

    for i, ele in enumerate(arr):
        value = float(str(value).replace(',', '.'))
        if ele > value:
            return i - 1

    return len(arr) - 1
