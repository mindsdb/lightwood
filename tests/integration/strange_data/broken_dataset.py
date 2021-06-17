from lightwood.api.types import DataAnalysis
from tests.integration.helpers import ClickhouseTest, break_dataset
from lightwood import analyze_dataset


class TestBrokenDatasets(ClickhouseTest):
    def get_ds(self, table, limit=300):
        from mindsdb_datasources import ClickhouseDS
        return ClickhouseDS(
            host=self.HOST,
            port=self.PORT,
            user=self.USER,
            password=self.PASSWORD,
            query='SELECT * FROM {}.{} LIMIT {}'.format(
                self.DATABASE,
                table,
                limit
            )
        )

    def test_broken_equals_normal(self):
        for dataset_name in ['home_rentals', 'hdi', 'us_health_insurance']:
            print(f'Trying dataset: {dataset_name}')
            ds = self.get_ds(dataset_name, limit=500)
            stats_1: DataAnalysis = analyze_dataset(ds)

            ds.df = break_dataset(ds.df)
            stats_2 = analyze_dataset(ds)

            for col in ds.df.columns:
                if col in stats_1.type_information.dtypes and col in stats_2.type_information.dtypes:
                    assert stats_1.type_information.dtypes[col] == stats_2.type_information.dtypes[col]
                else:
                    if not (col not in stats_1 and col not in stats_2):
                        raise AssertionError(f'Column {col} missing from analysis')