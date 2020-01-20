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

        X = []
        for row in data_source:
            X.append(np.array(row[0]))

        X = np.array(X)
        for target_col_name in self.targets:
            Y = data_source.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                self.targets[target_col_name]['model'] = xgb.XGBClassifier()
                self.targets[target_col_name]['model'].fit(X,Y)

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.NUMERIC:
                self.targets[target_col_name]['model'] = xgb.XGBRegressor()
                self.targets[target_col_name]['model'].fit(X,Y)

            else:
                self.targets[target_col_name]['model'] = None


    def predict(self, when_data_source, targets=None):
        X = []
        for row in when_data_source:
            X.append(np.array(row[0]))

        predictions = {}
        if targets is None:
            targets = self.targets
        for target_col_name in self.targets:
            if self.targets[target_col_name]['model'] is None:
                predictions[target_col_name] = None

            predictions[target_col_name] = self.targets[target_col_name]['model'].predict(X)

        return predictions
