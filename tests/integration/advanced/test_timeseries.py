import unittest
import numpy as np
import pandas as pd
from typing import List

from lightwood.api.types import ProblemDefinition

np.random.seed(0)


class TestTimeseries(unittest.TestCase):
    def check_ts_prediction_df(self, df: pd.DataFrame, nr_preds: int, orders: List[str]):
        for idx, row in df.iterrows():
            assert len(row['prediction']) == nr_preds

            for oby in orders:
                assert len(row[f'order_{oby}']) == nr_preds

            for t in range(nr_preds):
                assert row['lower'][t] <= row['prediction'][t] <= row['upper'][t]

            for oby in orders:
                assert len(row[f'order_{oby}']) == nr_preds

            if row['anomaly']:
                assert not (row['lower'][0] <= row['truth'] <= row['upper'][0])
            else:
                assert row['lower'][0] <= row['truth'] <= row['upper'][0]

    def test_0_grouped_regression_timeseries(self):
        """ Test grouped numerical predictions (forecast horizon > 1), covering most of the TS pipeline """
        from lightwood.api.high_level import predictor_from_problem

        data = pd.read_csv('tests/data/arrivals.csv')
        group = 'Country'
        train = pd.DataFrame(columns=data.columns)
        test = pd.DataFrame(columns=data.columns)

        train_ratio = 0.8
        for g in data[group].unique():
            subframe = data[data[group] == g]
            length = subframe.shape[0]
            train = train.append(subframe[:int(length * train_ratio)])
            test = test.append(subframe[int(length * train_ratio):])

        target = 'Traffic'
        nr_preds = 2
        order_by = 'T'

        # Test multiple predictors playing along together
        a = {}
        a[1] = predictor_from_problem(pd.read_csv('tests/data/hdi.csv'), ProblemDefinition.from_dict(
            {'target': 'Development Index', 'time_aim': 10}))
        a[1].learn(pd.read_csv('tests/data/hdi.csv'))
        a[1].predict(pd.read_csv('tests/data/hdi.csv').iloc[0:10])

        prdb = ProblemDefinition.from_dict({'target': target,
                                            'time_aim': 30,
                                            'nfolds': 10,
                                            'anomaly_detection': True,
                                            'timeseries_settings': {
                                                'use_previous_target': True,
                                                'group_by': ['Country'],
                                                'nr_predictions': nr_preds,
                                                'order_by': [order_by],
                                                'window': 5
                                            }
                                            })

        a[2] = predictor_from_problem(train, prdb)

        a[2].learn(train)
        preds = a[2].predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = a[2].predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # Additionally, check timestamps are further into the future than test dates
        latest_timestamp = pd.to_datetime(test[order_by]).max().timestamp()
        for idx, row in preds.iterrows():
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > latest_timestamp

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
                                                                        'nfolds': 5,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5
                                                                        },
                                                                        }))

        predictor.learn(train)
        predictor.predict(test)
