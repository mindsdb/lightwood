import unittest

class TestBasic(unittest.TestCase):
    def test_0_predict_file_flow(self):
        from lightwood import generate_predictor
        from mindsdb_datasources import FileDS

        datasource = FileDS('https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/adult_income/adult.csv')
        predictor = generate_predictor(datasource,'income')
        predictor.prepare(datasource)
        predictor.learn(datasource)

        predictions = predictor.predict(datasource)
        print(predictions[0:100])
