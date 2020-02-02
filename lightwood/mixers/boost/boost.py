import numpy as np
import xgboost as xgb

from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class BoostMixer():

    def __init__(self):
        self.targets = None

    def train(self, data_source):
        output_features = data_source.configuration['output_features']

        self.targets = {}
        for output_feature in output_features:
            self.targets[output_feature['name']] = {
                'type': output_feature['type']
            }
            if 'weights' in output_feature:
                self.targets[output_feature['name']]['weights'] = output_feature['weights']
            else:
                self.targets[output_feature['name']]['weights'] = None

        X = [[None] * len(data_source.configuration['input_features'])] * len(data_source)

        for feature_pos, input_feature in enumerate(data_source.configuration['input_features']):
            for i, point in enumerate(data_source.get_column_original_data(input_feature['name'])):
                X[i][feature_pos] = point

        X = np.array(X)

        for target_col_name in self.targets:
            Y = data_source.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1 for x in real]
                else:
                    sample_weight = []
                    for val in Y:
                        sample_weight.append(weight_map[val])

                self.targets[target_col_name]['model'] = xgb.XGBClassifier()
                self.targets[target_col_name]['model'].fit(X,Y,sample_weight=sample_weight)

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.NUMERIC:
                self.targets[target_col_name]['model'] = xgb.XGBRegressor()
                self.targets[target_col_name]['model'].fit(X,Y)

            else:
                self.targets[target_col_name]['model'] = None


    def predict(self, when_data_source, targets=None):

        X = [[None] * len(when_data_source.configuration['input_features'])] * len(when_data_source)

        for feature_pos, input_feature in enumerate(when_data_source.configuration['input_features']):
            for i, point in enumerate(when_data_source.get_column_original_data(input_feature['name'])):
                X[i][feature_pos] = point

        X = np.array(X)

        predictions = {}
        if targets is None:
            targets = self.targets
        for target_col_name in self.targets:
            if self.targets[target_col_name]['model'] is None:
                predictions[target_col_name] = None

            predictions[target_col_name] = self.targets[target_col_name]['model'].predict(X)

        return predictions
