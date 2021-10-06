import unittest
import numpy as np
import pandas as pd
from typing import List

from lightwood.api.types import ProblemDefinition
from lightwood.api.high_level import predictor_from_problem

np.random.seed(0)


class TestTimeseries(unittest.TestCase):
    def check_ts_prediction_df(self, df: pd.DataFrame, nr_preds: int, orders: List[str]):
        for idx, row in df.iterrows():
            lower = [row['lower']] if nr_preds == 1 else row['lower']
            upper = [row['upper']] if nr_preds == 1 else row['upper']
            prediction = [row['prediction']] if nr_preds == 1 else row['prediction']

            assert len(prediction) == nr_preds

            for oby in orders:
                assert len(row[f'order_{oby}']) == nr_preds

            for t in range(nr_preds):
                assert lower[t] <= prediction[t] <= upper[t]

            if row.get('anomaly', False):
                assert not (lower[0] <= row['truth'] <= upper[0])

    def split_arrivals(self, data: pd.DataFrame, grouped: bool) -> (pd.DataFrame, pd.DataFrame):
        train_ratio = 0.8

        if grouped:
            group = 'Country'
            train = pd.DataFrame(columns=data.columns)
            test = pd.DataFrame(columns=data.columns)
            for g in data[group].unique():
                subframe = data[data[group] == g]
                length = subframe.shape[0]
                train = train.append(subframe[:int(length * train_ratio)])
                test = test.append(subframe[int(length * train_ratio):])
        else:
            train = data[:int(data.shape[0] * train_ratio)]
            test = data[int(data.shape[0] * train_ratio):]

        return train, test

    def test_0_time_series_grouped_regression(self):
        """Test grouped numerical predictions, with anomalies and forecast horizon > 1 """
        data = pd.read_csv('tests/data/arrivals.csv')
        train, test = self.split_arrivals(data, grouped=True)
        target = 'Traffic'
        order_by = 'T'
        nr_preds = 2
        window = 5
        pred = predictor_from_problem(train,
                                      ProblemDefinition.from_dict({'target': target,
                                                                   'time_aim': 30,
                                                                   'anomaly_detection': True,
                                                                   'timeseries_settings': {
                                                                       'use_previous_target': True,
                                                                       'allow_incomplete_history': True,
                                                                       'group_by': ['Country'],
                                                                       'nr_predictions': nr_preds,
                                                                       'order_by': [order_by],
                                                                       'window': window
                                                                   }}))
        pred.learn(train)
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # test allowed incomplete history
        preds = pred.predict(test[:window - 1])
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # Additionally, check timestamps are further into the future than test dates
        latest_timestamp = pd.to_datetime(test[order_by]).max().timestamp()
        for idx, row in preds.iterrows():
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > latest_timestamp

    def test_1_time_series_regression(self):
        data = pd.read_csv('tests/data/arrivals.csv')
        train, test = self.split_arrivals(data, grouped=False)
        target = 'Traffic'
        order_by = 'T'
        nr_preds = 2
        window = 5
        pred = predictor_from_problem(data,
                                      ProblemDefinition.from_dict({'target': target,
                                                                   'anomaly_detection': False,
                                                                   'timeseries_settings': {
                                                                       'use_previous_target': False,
                                                                       'allow_incomplete_history': False,
                                                                       'nr_predictions': nr_preds,
                                                                       'order_by': [order_by],
                                                                       'window': window}
                                                                   }))
        pred.learn(data)
        preds = pred.predict(data[0:10])
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # test incomplete history, should not be possible
        self.assertRaises(Exception, pred.predict, test[:window - 1])

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

    def test_2_time_series_classification(self):
        from lightwood.api.high_level import predictor_from_problem

        df = pd.read_csv('tests/data/arrivals.csv')
        target = 'Traffic'
        df[target] = df[target] > 100000

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': target,
                                                                        'time_aim': 30,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5
                                                                        },
                                                                        }))

        predictor.learn(train)
        predictor.predict(test)
