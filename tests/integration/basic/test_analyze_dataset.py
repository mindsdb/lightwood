from lightwood import analyze_dataset
from lightwood.api import dtype
import unittest
from itertools import cycle
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tests.utils.data_generation import (
    test_column_types,
    generate_short_sentences,
    generate_rich_sentences,
    VOCAB,
)


class TestInferTypes(unittest.TestCase):
    def test_analyze_home_rentals(self):
        df = pd.read_csv(
            "https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/home_rentals/dataset/train.csv"
        )
        type_information = analyze_dataset(df).type_information

        self.assertTrue(
            type_information.dtypes["number_of_rooms"] == dtype.categorical)
        self.assertTrue(
            type_information.dtypes["number_of_bathrooms"] == dtype.binary)
        self.assertTrue(type_information.dtypes["sqft"] == dtype.integer)
        self.assertTrue(
            type_information.dtypes["location"] == dtype.categorical)
        self.assertTrue(
            type_information.dtypes["days_on_market"] == dtype.integer)
        self.assertTrue(
            type_information.dtypes["initial_price"] == dtype.integer)
        self.assertTrue(
            type_information.dtypes["neighborhood"] == dtype.categorical)
        # This has a .0 after every price, so we should detect it as float
        self.assertTrue(type_information.dtypes["rental_price"] == dtype.float)

        # There should be no ambiguity in the type detection for this dataset
        for k in type_information.additional_info:
            self.assertTrue(
                len(type_information.additional_info[k]["dtype_dist"]) == 1)

        # None of the columns are identifiers
        for k in type_information.identifiers:
            self.assertTrue(type_information.identifiers[k] is None)

    def test_with_generated(self):
        # Must be an even number
        n_points = 134

        # Apparently for n_category_values = 10 it doesn't work
        n_category_values = 4
        categories_cycle = cycle(range(n_category_values))
        n_multilabel_category_values = 25
        multiple_categories_str_cycle = cycle(
            random.choices(VOCAB[0:20], k=n_multilabel_category_values)
        )

        df = pd.DataFrame(
            {
                "numeric_int": [x % 10 for x in list(range(n_points))],
                "numeric_float": np.linspace(0, n_points, n_points),
                "date_timestamp": [
                    (datetime.now() - timedelta(minutes=int(i))).isoformat()
                    for i in range(n_points)
                ],
                "date_date": [
                    (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(n_points)
                ],
                "categorical_str": [
                    f"category_{next(categories_cycle)}" for i in range(n_points)
                ],
                "categorical_int": [next(categories_cycle) for i in range(n_points)],
                "categorical_binary": [0, 1] * (n_points // 2),
                "sequential_numeric_array": [f"1,2,3,4,5,{i}" for i in range(n_points)],
                "multiple_categories_array_str": [
                    ",".join(
                        [
                            f"{next(multiple_categories_str_cycle)}"
                            for j in range(random.randint(1, 6))
                        ]
                    )
                    for i in range(n_points)
                ],
                "short_text": generate_short_sentences(n_points),
                "rich_text": generate_rich_sentences(n_points),
            }
        )

        analysis = analyze_dataset(df)
        type_information = analysis.type_information
        for col_name in df.columns:
            expected_type = test_column_types[col_name]
            print(
                f"Got {type_information.dtypes[col_name]} | Expected: {expected_type}"
            )
            assert type_information.dtypes[col_name] == expected_type

        stats = analysis.statistical_analysis
        for k in stats.histograms:
            if k != 'sequential_numeric_array':
                assert np.sum(stats.histograms[k]['y']) == n_points

        assert set(stats.histograms['short_text']['x']) == set(['Unknown'])
        assert set(stats.histograms['rich_text']['x']) == set(['Unknown'])
        assert set(stats.histograms['rich_text']['x']) == set(['Unknown'])
        # 50 is a magic number, when we change this, tests must change
        assert len(set(stats.histograms['date_timestamp']['x'])) == 50
        assert len(set(stats.histograms['date_date']['x'])) == 50


'''
    # These are native tests, we should adapt them to the new lightwood at some point.
    # Not critical since we removed a bunch of these capacities, other are well tested in the new test above.
    def test_deduce_foreign_key(self):
        """Tests that basic cases of type deduction work correctly"""
        predictor = Predictor(name='test_deduce_foreign_key')
        predictor.breakpoint = 'DataAnalyzer'

        n_points = 100

        df = pd.DataFrame({
            'numeric_id': list(range(n_points)),
            'uuid': [str(uuid4()) for i in range(n_points)],
            'to_predict': [i % 5 for i in range(n_points)]
        })

        try:
            predictor.learn(from_data=df, to_predict='to_predict')
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert isinstance(stats_v2['numeric_id']['identifier'], str)
        assert isinstance(stats_v2['uuid']['identifier'], str)

        assert 'numeric_id' in predictor.transaction.lmd['columns_to_ignore']
        assert 'uuid' in predictor.transaction.lmd['columns_to_ignore']

    def test_empty_values(self):
        predictor = Predictor(name='test_empty_values')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[::2] = None

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 50
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 50

    def test_type_mix(self):
        predictor = Predictor(name='test_type_mix')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_float_1': np.linspace(0, n_points, n_points),
            'numeric_float_2': np.linspace(0, n_points, n_points),
            'numeric_float_3': np.linspace(0, n_points, n_points),
        })
        df['numeric_float_1'].iloc[:2] = 'random string'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_float_3',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError


        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_float_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_float_1']['typing']['data_subtype'] == DATA_SUBTYPES.FLOAT
        assert stats_v2['numeric_float_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == 98
        assert stats_v2['numeric_float_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.FLOAT] == 98

    def test_sample(self):
        sample_settings = {
            'sample_for_analysis': True,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_1')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 100
        df = pd.DataFrame({
            'numeric_int_1': [x % 10 for x in list(range(n_points))],
            'numeric_int_2': [x % 10 for x in list(range(n_points))]
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']
        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] <= n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] <= n_points

        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': sample_data
        }
        sample_settings['sample_function'] = mock.MagicMock(wraps=sample_data)
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_sample_2')
        predictor.breakpoint = 'TypeDeductor'

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

    def test_small_dataset_no_sampling(self):
        sample_settings = {
            'sample_for_analysis': False,
            'sample_function': mock.MagicMock(wraps=sample_data)
        }
        setattr(sample_settings['sample_function'], '__name__', 'sample_data')

        predictor = Predictor(name='test_small_dataset_no_sampling')
        predictor.breakpoint = 'TypeDeductor'

        n_points = 50
        df = pd.DataFrame({
            'numeric_int_1': [*range(n_points)],
            'numeric_int_2': [*range(n_points)],
        })

        try:
            predictor.learn(
                from_data=df,
                to_predict='numeric_int_2',
                advanced_args={'force_column_usage': list(df.columns)},
                sample_settings=sample_settings
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert not sample_settings['sample_function'].called

        stats_v2 = predictor.transaction.lmd['stats_v2']

        assert stats_v2['numeric_int_1']['typing']['data_type'] == DATA_TYPES.NUMERIC
        assert stats_v2['numeric_int_1']['typing']['data_subtype'] == DATA_SUBTYPES.INT

        # This ensures that no sampling was applied
        assert stats_v2['numeric_int_1']['typing']['data_type_dist'][DATA_TYPES.NUMERIC] == n_points
        assert stats_v2['numeric_int_1']['typing']['data_subtype_dist'][DATA_SUBTYPES.INT] == n_points

    def test_date_formats(self):
        n_points = 50
        df = pd.DataFrame({
            'date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n_points)],
            'datetime': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%dT%H:%M') for i in range(n_points)],
        })

        predictor = Predictor(name='test_date_formats')
        predictor.breakpoint = 'TypeDeductor'

        try:
            predictor.learn(
                from_data=df,
                to_predict='datetime',
                advanced_args={'force_column_usage': list(df.columns)}
            )
        except BreakpointException:
            pass
        else:
            raise AssertionError

        assert predictor.transaction.lmd['stats_v2']['date']['typing']['data_type'] == DATA_TYPES.DATE
        assert predictor.transaction.lmd['stats_v2']['date']['typing']['data_subtype'] == DATA_SUBTYPES.DATE

        assert predictor.transaction.lmd['stats_v2']['datetime']['typing']['data_type'] == DATA_TYPES.DATE
        assert predictor.transaction.lmd['stats_v2']['datetime']['typing']['data_subtype'] == DATA_SUBTYPES.TIMESTAMP
'''
