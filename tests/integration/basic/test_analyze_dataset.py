from lightwood import analyze_dataset
from type_infer.dtype import dtype
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
