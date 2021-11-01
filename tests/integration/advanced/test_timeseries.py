import unittest
import numpy as np
import pandas as pd
import time
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

    def calculate_duration(self, predictor, train, time_aim_expected):

        start = time.process_time()
        predictor.learn(train)
        time_aim_actual = (time.process_time() - start)
        if((time_aim_expected * 5) < time_aim_actual):
            error = 'time_aim is set to {} seconds, however learning took {}'.format(time_aim_expected, time_aim_actual)
            raise ValueError(error)
        assert (time_aim_expected * 5) >= time_aim_actual
        return predictor

    def test_0_time_series_grouped_regression(self):
        """Test grouped numerical predictions, with anomalies and forecast horizon > 1 """
        data = pd.read_csv('tests/data/arrivals.csv')
        train, test = self.split_arrivals(data, grouped=True)
        target = 'Traffic'
        time_aim_expected = 30
        order_by = 'T'
        nr_preds = 2
        window = 5
        pred = predictor_from_problem(train,
                                      ProblemDefinition.from_dict({'target': target,
                                                                   'time_aim': time_aim_expected,
                                                                   'anomaly_detection': True,
                                                                   'timeseries_settings': {
                                                                       'use_previous_target': True,
                                                                       'allow_incomplete_history': True,
                                                                       'group_by': ['Country'],
                                                                       'nr_predictions': nr_preds,
                                                                       'order_by': [order_by],
                                                                       'window': window
                                                                   }}))
        pred = self.calculate_duration(pred, train, time_aim_expected)
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

        # Check custom ICP params
        test.pop('__mdb_make_predictions')
        preds = pred.predict(test, {'fixed_confidence': 0.01, 'anomaly_cooldown': 100, 'anomaly_error_rate': 1})
        assert all([all([v == 0.01 for v in f]) for f in preds['confidence'].values])
        assert pred.pred_args.anomaly_error_rate == 1
        assert pred.pred_args.anomaly_cooldown == 100

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
        time_aim_expected = 30
        df[target] = df[target] > 100000

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': target,
                                                                        'time_aim': time_aim_expected,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5
                                                                        },
                                                                        }))

        predictor = self.calculate_duration(predictor, train, time_aim_expected)
        predictor.predict(test)

    def test_3_time_series_sktime_mixer(self):
        """
        Tests `sktime` mixer individually, as it has a special notion of absolute
        temporal timestamps that we need to ensure are being used correctly. In
        particular, given a train-dev-test split, any forecasts coming from a sktime
        mixer should start from the latest observed data in the entire dataset.
        """  # noqa

        from sklearn.metrics import r2_score
        from scipy import signal
        from lightwood.api.high_level import (
            ProblemDefinition,
            json_ai_from_problem,
            code_from_json_ai,
            predictor_from_code,
        )

        # synth square wave
        tsteps = 100
        target = 'Value'
        t = np.linspace(0, 1, tsteps, endpoint=False)
        ts = signal.sawtooth(2 * np.pi * 5 * t, width=0.5)
        df = pd.DataFrame(columns=['Time', target])
        df['Time'] = t
        df[target] = ts

        train = df[:int(len(df) * 0.8)]
        test = df[int(len(df) * 0.8):]

        pdef = ProblemDefinition.from_dict({'target': target,
                                            'time_aim': 10,
                                            'timeseries_settings': {
                                                'order_by': ['Time'],
                                                'window': 5,
                                                'nr_predictions': 20
                                            }})

        json_ai = json_ai_from_problem(df, problem_definition=pdef)
        json_ai.outputs[target].mixers = [{
            "module": "SkTime",
            "args": {
                "stop_after": "$problem_definition.seconds_per_mixer",
                "n_ts_predictions": "$problem_definition.timeseries_settings.nr_predictions",
            }}]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        predictor.learn(train)
        ps = predictor.predict(test)

        assert r2_score(ps['truth'].values, ps['prediction'].iloc[0]) >= 0.95
