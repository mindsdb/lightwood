import unittest
import pandas as pd
from typing import List
from datetime import datetime

from lightwood.api.types import ProblemDefinition


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

    def test_grouped_regression_timeseries(self):
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

        predictor = predictor_from_problem(train, ProblemDefinition.from_dict({'target': target,
            'time_aim': 30,
            'nfolds': 10,
            'anomaly_detection': True,
            'timeseries_settings': {
                'use_previous_target': True,
                'group_by': ['Country'],
                'nr_predictions': nr_preds,
                'order_by': [order_by],
                'window': 5
            },
            }))

        predictor.learn(train)
        preds = predictor.predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = predictor.predict(test)
        self.check_ts_prediction_df(preds, nr_preds, [order_by])

        # Additionally, check timestamps are further into the future than test dates
        latest_timestamp = pd.to_datetime(test[order_by]).max().timestamp()
        for idx, row in preds.iterrows():
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > latest_timestamp

    @unittest.skip
    def test_time_series_classification(self):
        from lightwood.api.high_level import predictor_from_problem

        datasource = pd.read_csv('tests/data/occupancy.csv')  # @TODO: make into synth dataset for faster execution
        target = 'Occupancy'

        predictor = predictor_from_problem(
        datasource.df, ProblemDefinition.from_dict({'target': target,
                'time_aim': 100,
                'nfolds': 10,
                'anomaly_detection': False,
                'timeseries_settings': {
                    'order_by': ['date'],
                    'use_previous_target': True,
                    'window': 10
                },
                }))

        predictor.learn(datasource.df)
        predictions = predictor.predict(datasource.df)
