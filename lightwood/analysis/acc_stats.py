import random
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from lightwood.api.dtype import dtype


class AccStats:
    """
    Computes accuracy stats and a confusion matrix for the validation dataset
    """

    def __init__(self, dtype_dict: dict, target: str, buckets: Union[None, dict]):
        self.col_stats = dtype_dict
        self.target = target
        self.input_cols = list(dtype_dict.keys())
        self.buckets = buckets if buckets else {}

        self.normal_predictions_bucketized = []
        self.real_values_bucketized = []
        self.numerical_samples_arr = []

    def fit(self, input_df: pd.DataFrame, predictions: pd.DataFrame, conf=Union[None, np.ndarray]):
        column_indexes = {}
        for i, col in enumerate(self.input_cols):
            column_indexes[col] = i

        real_present_inputs_arr = []
        for _, row in input_df.iterrows():
            present_inputs = [1] * len(self.input_cols)
            for i, col in enumerate(self.input_cols):
                if str(row[col]) in ('None', 'nan', '', 'Nan', 'NAN', 'NaN'):
                    present_inputs[i] = 0
            real_present_inputs_arr.append(present_inputs)

        for n in range(len(predictions)):
            row = input_df.iloc[n]
            real_value = row[self.target]
            predicted_value = predictions.iloc[n]['prediction']

            if isinstance(predicted_value, list):
                # T+N time series, for now we compare the T+1 prediction only @TODO: generalize
                predicted_value = predicted_value[0]

            predicted_value = predicted_value \
                if self.col_stats[self.target] not in [dtype.integer, dtype.float] \
                else float(predicted_value)

            real_value = real_value \
                if self.col_stats[self.target] not in [dtype.integer, dtype.float] \
                else float(real_value)

            if self.buckets:
                bucket = self.buckets[self.target]
                predicted_value_b = get_value_bucket(predicted_value, bucket, self.col_stats[self.target])
                real_value_b = get_value_bucket(real_value, bucket, self.col_stats[self.target])
            else:
                predicted_value_b = predicted_value
                real_value_b = real_value

            if conf is not None and self.col_stats[self.target] in [dtype.integer, dtype.float]:
                predicted_range = conf.iloc[n][['lower', 'upper']].tolist()
            else:
                predicted_range = (predicted_value_b, predicted_value_b)

            self.real_values_bucketized.append(real_value_b)
            self.normal_predictions_bucketized.append(predicted_value_b)
            if conf is not None and self.col_stats[self.target] in [dtype.integer, dtype.float]:
                self.numerical_samples_arr.append((real_value, predicted_range))

    def get_accuracy_stats(self):
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

        overall_accuracy = sum(accuracy_count) / len(accuracy_count)

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
            'accuracies': list(bucket_accuracy.values())
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

    elif target_dtype in (dtype.integer, dtype.float):
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
