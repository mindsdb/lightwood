import xgboost as xgb


class BoostMixer():

    def __init__(self):
        pass

    def fit(self, data_source):
        input_features = data_source.configuration['input_features']
        output_features data_source.configuration['output_features']

        print(input_features)
        print(output_features)

        for row in data_source:
            print(row)

            
    def predict(self, when_data_source):
        pass
