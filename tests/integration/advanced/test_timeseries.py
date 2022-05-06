import random
import unittest
import numpy as np
import pandas as pd
from typing import List
from scipy import signal
from sklearn.metrics import r2_score
from lightwood.api.types import ProblemDefinition
from tests.utils.timing import train_and_check_time_aim
from sktime.forecasting.base import ForecastingHorizon
try:
    from sktime.forecasting.statsforecast import StatsForecastAutoARIMA as AutoARIMA
except ModuleNotFoundError:
    from sktime.forecasting.arima import AutoARIMA

from lightwood.api.high_level import json_ai_from_problem, code_from_json_ai, predictor_from_code, predictor_from_problem  # noqa
from lightwood.mixer.sktime import SkTime

np.random.seed(0)


class TestTimeseries(unittest.TestCase):
    def check_ts_prediction_df(self, df: pd.DataFrame, horizon: int, orders: List[str]):
        for idx, row in df.iterrows():
            lower = [row['lower']] if horizon == 1 else row['lower']
            upper = [row['upper']] if horizon == 1 else row['upper']
            prediction = [row['prediction']] if horizon == 1 else row['prediction']

            assert len(prediction) == horizon

            for oby in orders:
                assert len(row[f'order_{oby}']) == horizon
                assert not any(pd.isna(row[f'order_{oby}']))

            for t in range(horizon):
                assert lower[t] <= prediction[t] <= upper[t]

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
        horizon = 2
        window = 5
        jai = json_ai_from_problem(train,
                                   ProblemDefinition.from_dict({'target': target,
                                                                'time_aim': 30,
                                                                'anomaly_detection': True,
                                                                'timeseries_settings': {
                                                                    'use_previous_target': True,
                                                                    'allow_incomplete_history': True,
                                                                    'group_by': ['Country'],
                                                                    'horizon': horizon,
                                                                    'order_by': [order_by],
                                                                    'period_intervals': (('daily', 7),),
                                                                    'window': window
                                                                }}))
        for i, mixer in enumerate(jai.model['args']['submodels']):
            if mixer["module"] == 'SkTime':
                sktime_mixer_idx = i

        jai.model['args']['submodels'][sktime_mixer_idx] = {
            "module": "SkTime",
            "args": {
                "stop_after": "$problem_definition.seconds_per_mixer",
                "horizon": "$problem_definition.timeseries_settings.horizon",
                "model_path": "'trend.TrendForecaster'",  # use a cheap forecaster
                "hyperparam_search": False,  # disable this as it's expensive and covered in test #3
            },
        }

        code = code_from_json_ai(jai)
        pred = predictor_from_code(code)

        # Test with a short time aim
        train_and_check_time_aim(pred, train)
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, horizon, [order_by])

        # test allowed incomplete history
        preds = pred.predict(test[:window - 1])
        self.check_ts_prediction_df(preds, horizon, [order_by])

        # test inferring mode
        test['__mdb_make_predictions'] = False
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, horizon, [order_by])

        # Additionally, check timestamps are further into the future than test dates
        latest_timestamp = pd.to_datetime(test[order_by]).max().timestamp()
        for idx, row in preds.iterrows():
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > latest_timestamp

        # Check custom ICP params
        test.pop('__mdb_make_predictions')
        preds = pred.predict(test, {'fixed_confidence': 0.01, 'anomaly_cooldown': 100})
        assert all([all([v == 0.01 for v in f]) for f in preds['confidence'].values])
        assert pred.pred_args.anomaly_cooldown == 100

    def test_1_time_series_regression(self):
        np.random.seed(0)
        data = pd.read_csv('tests/data/arrivals.csv')
        train_df, test_df = self.split_arrivals(data, grouped=False)
        target = 'Traffic'
        order_by = 'T'
        horizon = 2
        window = 5
        pred = predictor_from_problem(data,
                                      ProblemDefinition.from_dict({'target': target,
                                                                   'anomaly_detection': False,
                                                                   'timeseries_settings': {
                                                                       'use_previous_target': False,
                                                                       'allow_incomplete_history': False,
                                                                       'horizon': horizon,
                                                                       'order_by': [order_by],
                                                                       'window': window}
                                                                   }))

        # add a few invalid datetime values to test cleaning procedures
        for idx in list(np.where(np.random.random((len(train_df),)) > 0.98)[0]):
            train_df.at[idx, 'T'] = pd.NaT

        pred.learn(train_df)
        preds = pred.predict(data.sample(frac=1)[0:10])
        self.assertTrue('original_index' in preds.columns)
        self.check_ts_prediction_df(preds, horizon, [order_by])

        # test incomplete history, should not be possible
        self.assertRaises(Exception, pred.predict, test_df[:window - 1])

        # test inferring mode
        test_df['__mdb_make_predictions'] = False
        test_df = test_df.sample(frac=1)  # shuffle to test internal ordering logic
        preds = pred.predict(test_df)
        self.check_ts_prediction_df(preds, horizon, [order_by])

        # Additionally, check timestamps are further into the future than test dates
        latest_timestamp = pd.to_datetime(test_df[order_by]).max().timestamp()
        for idx, row in preds.iterrows():
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > latest_timestamp

    def test_2_time_series_classification_short_horizon_binary(self):
        df = pd.read_csv('tests/data/arrivals.csv')[:127]
        target = 'Traffic'
        df[target] = df[target] > 100000

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': target,
                                                                        'time_aim': 80,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5
                                                                        },
                                                                        }))

        predictor.learn(train)
        predictor.predict(test)

    def test_3_time_series_classification_long_horizon_binary(self):
        df = pd.read_csv('tests/data/arrivals.csv')[:127]
        target = 'Traffic'
        df[target] = df[target] > 100000

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': target,
                                                                        'time_aim': 80,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5,
                                                                            'horizon': 2
                                                                        },
                                                                        }))

        predictor.learn(train)
        predictor.predict(test)

    def test_4_time_series_classification_long_horizon_multiclass(self):
        df = pd.read_csv('tests/data/arrivals.csv')[:127]  # enforce "Country" to be "No information"
        target = 'Traffic'
        df[target] = df[target].apply(lambda x: chr(65 + int(str(x / 10000)[0])))  # multiclass time series target

        # test array columns as additional input
        df['test_num_array'] = [[random.choice([1, 2, 3, 4]) for __ in range(4)] for _ in range(df.shape[0])]
        df['test_cat_array'] = [[random.choice(['a', 'b', 'c', 'd']) for __ in range(4)] for _ in range(df.shape[0])]

        train_idxs = np.random.rand(len(df)) < 0.8
        train = df[train_idxs]
        test = df[~train_idxs]

        predictor = predictor_from_problem(df,
                                           ProblemDefinition.from_dict({'target': target,
                                                                        'time_aim': 80,
                                                                        'anomaly_detection': False,
                                                                        'timeseries_settings': {
                                                                            'order_by': ['T'],
                                                                            'use_previous_target': True,
                                                                            'window': 5,
                                                                            'horizon': 2
                                                                        },
                                                                        }))

        predictor.learn(train)
        predictor.predict(test)

    def test_5_time_series_sktime_mixer(self):
        """
        Tests `sktime` mixer individually, as it has a special notion of
        timestamps that we need to ensure are being used correctly. In
        particular, given a train-dev-test split, any forecasts coming from a sktime
        mixer should start from the latest observed data in the entire dataset.
        
        This test also compares against manual use of sktime to ensure equal results.
        """  # noqa

        # synth square wave
        tsteps = 100
        target = 'Value'
        horizon = 20
        t = np.linspace(0, 1, tsteps, endpoint=False)
        ts = [i + f for i, f in enumerate(signal.sawtooth(2 * np.pi * 5 * t, width=0.5))]
        df = pd.DataFrame(columns=['Time', target])
        df['Time'] = t
        df[target] = ts
        df[f'{target}_2x'] = [2 * elt for elt in ts]

        train = df[:int(len(df) * 0.8)]
        test = df[int(len(df) * 0.8):]

        pdef = ProblemDefinition.from_dict({'target': target,
                                            'time_aim': 200,
                                            'timeseries_settings': {
                                                'order_by': ['Time'],
                                                'window': 5,
                                                'horizon': horizon,
                                                'historical_columns': [f'{target}_2x']
                                            }})

        json_ai = json_ai_from_problem(df, problem_definition=pdef)
        json_ai.model['args']['submodels'] = [{
            "module": "SkTime",
            "args": {
                "stop_after": "$problem_definition.seconds_per_mixer",
                "horizon": "$problem_definition.timeseries_settings.horizon",
            }}]

        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        # Test with a longer time aim
        train_and_check_time_aim(predictor, train)
        ps = predictor.predict(test)
        assert r2_score(test[target].values, ps['prediction'].iloc[0]) >= 0.95

        # test historical columns asserts
        test[f'{target}_2x'].iloc[0] = np.nan
        self.assertRaises(Exception, predictor.predict, test)

        test.pop(f'{target}_2x')
        self.assertRaises(Exception, predictor.predict, test)

        # compare vs sktime manual usage
        if isinstance(predictor.ensemble.mixers[predictor.ensemble.best_index], SkTime):
            forecaster = AutoARIMA()
            fh = ForecastingHorizon([i for i in range(int(tsteps * 0.8))], is_relative=True)
            forecaster.fit(train[target], fh=fh)
            manual_preds = forecaster.predict(fh[1:horizon + 1]).tolist()
            lw_preds = [p[0] for p in ps['prediction']]
            assert np.allclose(manual_preds, lw_preds, atol=1)

    def test_6_irregular_series(self):
        """
        Even though the suggestion is to feed regularly sampled series into predictors, this test can still help us
        catch undesired behavior when the recommendation is not followed.
        """  # noqa

        # synth square wave
        tsteps = 100
        target = 'Value'
        horizon = 20
        # added random noise for irregular sampling
        np.random.seed(0)
        t = np.linspace(0, 1, tsteps, endpoint=False) + np.random.uniform(size=(tsteps,), low=-0.005, high=0.005)
        ts = [i + f for i, f in enumerate(signal.sawtooth(2 * np.pi * 5 * t, width=0.5))]
        df = pd.DataFrame(columns=['Time', target])
        df['Time'] = t
        df[target] = ts
        df[f'{target}_2x'] = [2 * elt for elt in ts]

        train = df[:int(len(df) * 0.8)]
        test = df[int(len(df) * 0.8):]

        pdef = ProblemDefinition.from_dict({'target': target,
                                            'time_aim': 200,
                                            'timeseries_settings': {
                                                'order_by': ['Time'],
                                                'window': 5,
                                                'horizon': horizon,
                                                'historical_columns': [f'{target}_2x']
                                            }})

        json_ai = json_ai_from_problem(df, problem_definition=pdef)
        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        train_and_check_time_aim(predictor, train)  # Test with a longer time aim

        test['__mdb_make_predictions'] = False
        predictor.predict(test)
