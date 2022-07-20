import random
from datetime import datetime
import unittest
import numpy as np
import pandas as pd
from typing import List
from scipy import signal
from sklearn.metrics import r2_score
from lightwood.api.types import ProblemDefinition
from tests.utils.timing import train_and_check_time_aim
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA as AutoARIMA

from lightwood.api.high_level import json_ai_from_problem, code_from_json_ai, predictor_from_code, predictor_from_problem  # noqa
from lightwood.data.splitter import stratify
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
                row[f'order_{oby}'] = [row[f'order_{oby}']] if horizon == 1 else row[f'order_{oby}']
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
        window = 5

        for horizon in (1, 2):
            jai = json_ai_from_problem(train,
                                       ProblemDefinition.from_dict({'target': target,
                                                                    'time_aim': 30,
                                                                    'anomaly_detection': True,
                                                                    'timeseries_settings': {
                                                                        'use_previous_target': True,
                                                                        'allow_incomplete_history': True,
                                                                        'group_by': ['Country'],
                                                                        'horizon': horizon,
                                                                        'order_by': order_by,
                                                                        'period_intervals': (('daily', 7),),
                                                                        'window': window
                                                                    }}))
            sktime_mixer_idx = None
            for i, mixer in enumerate(jai.model['args']['submodels']):
                if mixer["module"] == 'SkTime':
                    sktime_mixer_idx = i

            if sktime_mixer_idx:
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

            # test inferring mode, check timestamps are further into the future than test dates
            test['__mdb_forecast_offset'] = 1
            preds = pred.predict(test)
            self.check_ts_prediction_df(preds, horizon, [order_by])

            latest_timestamp = pd.to_datetime(test[order_by]).max().timestamp()
            for idx, row in preds.iterrows():
                row[f'order_{order_by}'] = [row[f'order_{order_by}']] if horizon == 1 else row[f'order_{order_by}']
                for timestamp in row[f'order_{order_by}']:
                    assert timestamp > latest_timestamp

            # Check custom ICP params
            test.pop('__mdb_forecast_offset')
            preds = pred.predict(test, {'fixed_confidence': 0.01, 'anomaly_cooldown': 100})
            if horizon == 1:
                assert set([v for v in preds['confidence'].values]) == {0.01}
            else:
                assert all([all([v == 0.01 for v in f]) for f in preds['confidence'].values])
            assert pred.pred_args.anomaly_cooldown == 100

    def test_1_time_series_regression(self):
        np.random.seed(0)
        data = pd.read_csv('tests/data/arrivals.csv')
        data = data[data['Country'] == 'US']
        train_df, test_df = self.split_arrivals(data, grouped=False)
        target = 'Traffic'
        order_by = 'T'
        window = 5
        for horizon in (1, 2):
            jai = json_ai_from_problem(data,
                                       ProblemDefinition.from_dict({'target': target,
                                                                    'anomaly_detection': False,
                                                                    'timeseries_settings': {
                                                                        'use_previous_target': False,
                                                                        'allow_incomplete_history': False,
                                                                        'horizon': horizon,
                                                                        'order_by': order_by,
                                                                        'window': window}
                                                                    }))
            jai.model['args']['submodels'] = [jai.model['args']['submodels'][0]]
            code = code_from_json_ai(jai)
            pred = predictor_from_code(code)

            # add a few invalid datetime values to test cleaning procedures
            for idx in list(np.where(np.random.random((len(train_df),)) > 0.98)[0]):
                train_df.at[idx, 'T'] = pd.NaT

            pred.learn(train_df)
            preds = pred.predict(data.sample(frac=1)[0:10])
            self.assertTrue('original_index' in preds.columns)
            self.check_ts_prediction_df(preds, horizon, [order_by])

            # test incomplete history, should not be possible
            self.assertRaises(Exception, pred.predict, test_df[:window - 1])

            # test inferring mode, check timestamps are further into the future than test dates
            test_df['__mdb_forecast_offset'] = 1
            test_df = test_df.sample(frac=1)  # shuffle to test internal ordering logic
            preds = pred.predict(test_df)
            self.check_ts_prediction_df(preds, horizon, [order_by])

            latest_timestamp = pd.to_datetime(test_df[order_by]).max().timestamp()
            for idx, row in preds.iterrows():
                row[f'order_{order_by}'] = [row[f'order_{order_by}']] if horizon == 1 else row[f'order_{order_by}']
                for timestamp in row[f'order_{order_by}']:
                    assert timestamp > latest_timestamp

            # test null offset mode
            test_df['__mdb_forecast_offset'] = 0
            preds = pred.predict(test_df)
            self.check_ts_prediction_df(preds, horizon, [order_by])
            assert preds.shape[0] == 1
            if horizon == 1:
                last_dt = datetime.utcfromtimestamp(preds[f'order_{order_by}'].values[0])
            else:
                last_dt = datetime.utcfromtimestamp(preds[f'order_{order_by}'].values[0][0])
            formatted = str(last_dt.year) + '-' + str(last_dt.month)
            assert formatted == test_df.sort_values(by=order_by).iloc[-1][order_by]

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
                                                                            'order_by': 'T',
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
                                                                            'order_by': 'T',
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
                                                                            'order_by': 'T',
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
        
        This test also compares:
         - correct propagation of offset by K if the special `__mdb_forecast_offset` column is present
         - results against manual use of sktime to ensure equal results.
        """  # noqa

        # synth square wave
        tsteps = 100
        target = 'Value'
        horizon = 20
        t = np.linspace(0, 100, tsteps, endpoint=False)
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
                                                'order_by': 'Time',
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

        # test offset for infer mode
        test['__mdb_forecast_offset'] = 1  # one step after latest (inferred)
        ps1 = predictor.predict(test)
        test['__mdb_forecast_offset'] = 0  # at latest
        ps0 = predictor.predict(test)
        test['__mdb_forecast_offset'] = -1  # one step before latest
        psm1 = predictor.predict(test)
        times_1 = psm1['order_Time'].tolist()[0]
        values_1 = psm1['prediction'].tolist()[0]
        times0 = ps0['order_Time'].tolist()[0]
        values0 = ps0['prediction'].tolist()[0]
        times1 = ps1['order_Time'].tolist()[0]
        values1 = ps1['prediction'].tolist()[0]

        # due to the offset, these intermediate indexes should be equal
        self.assertTrue(times_1[1:] == times0[0:-1])
        self.assertTrue(times0[1:] == times1[0:-1])
        self.assertTrue(values_1[1:] == values0[0:-1])
        self.assertTrue(values0[1:] == values1[0:-1])

        # the rest should be different
        self.assertTrue(times_1 != times0)
        self.assertTrue(times0 != times1)
        self.assertTrue(values_1 != values0)
        self.assertTrue(values0 != values1)

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
            assert np.allclose(manual_preds, lw_preds, atol=1.5)

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
        t = np.linspace(0, 100, tsteps, endpoint=False) + np.random.uniform(size=(tsteps,), low=-0.005, high=0.005)
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
                                                'order_by': 'Time',
                                                'window': 5,
                                                'horizon': horizon,
                                                'historical_columns': [f'{target}_2x']
                                            }})

        json_ai = json_ai_from_problem(df, problem_definition=pdef)
        code = code_from_json_ai(json_ai)
        predictor = predictor_from_code(code)

        train_and_check_time_aim(predictor, train)  # Test with a longer time aim

        test['__mdb_forecast_offset'] = 1
        predictor.predict(test)

    def test_7_time_series_double_grouped_regression(self):
        """Test double-grouped numerical predictions, replicates quick start guide in cloud.mindsdb.com """
        data = pd.read_csv('tests/data/house_sales.csv')
        gby = ['bedrooms', 'type']
        target = 'MA'
        order_by = 'saledate'
        window = 8
        horizon = 4
        train, _, test = stratify(data, pct_train=0.8, pct_dev=0, pct_test=0.2, stratify_on=gby, seed=1,
                                  reshuffle=False)
        jai = json_ai_from_problem(train,
                                   ProblemDefinition.from_dict({'target': target,
                                                                'time_aim': 30,
                                                                'timeseries_settings': {
                                                                    'group_by': gby,
                                                                    'horizon': horizon,
                                                                    'order_by': order_by,
                                                                    'window': window
                                                                }}))
        code = code_from_json_ai(jai)
        pred = predictor_from_code(code)

        # Test with a short time aim with inferring mode, check timestamps are further into the future than test dates
        test['__mdb_forecast_offset'] = 1
        train_and_check_time_aim(pred, train)
        preds = pred.predict(test)
        self.check_ts_prediction_df(preds, horizon, [order_by])

        for idx, row in preds.iterrows():
            row[f'order_{order_by}'] = [row[f'order_{order_by}']] if horizon == 1 else row[f'order_{order_by}']
            for timestamp in row[f'order_{order_by}']:
                assert timestamp > pd.to_datetime(test[order_by]).max().timestamp()
