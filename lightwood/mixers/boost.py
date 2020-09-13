import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import BaseMixer


class BoostMixer(BaseMixer):
    def __init__(self, quantiles=None):
        super().__init__()
        self.targets = None
        self.quantiles = quantiles

    def _row_to_ndarray(self, row):
        sample = []
        for feature in row['input_features'].values():
            sample.extend(feature)
        return np.array(sample)

    def fit(self, train_ds, test_ds):
        output_features = train_ds.configuration['output_features']

        self.targets = {}
        
        for output_feature in output_features:
            self.targets[output_feature['name']] = {
                'type': output_feature['type']
            }
            if 'weights' in output_feature:
                self.targets[output_feature['name']]['weights'] = output_feature['weights']
            else:
                self.targets[output_feature['name']]['weights'] = None

        X = []
        for row in train_ds:
            X.append(self._row_to_ndarray(row))

        for target_col_name in self.targets:
            Y = train_ds.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                sample_weight = [1 for _ in X]
                if weight_map is None:
                    sample_weight = [1] * len(Y)
                else:
                    sample_weight = []
                    for val in Y:
                        sample_weight.append(weight_map[val])

                self.targets[target_col_name]['model'] = GradientBoostingClassifier(n_estimators=600)
                self.targets[target_col_name]['model'].fit(X, Y, sample_weight=sample_weight)

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.NUMERIC:
                self.targets[target_col_name]['model'] = GradientBoostingRegressor(n_estimators=600)
                self.targets[target_col_name]['model'].fit(X, Y)
                if self.quantiles is not None:
                    self.targets[target_col_name]['quantile_models'] = {}
                    for i, quantile in enumerate(self.quantiles):
                        self.targets[target_col_name]['quantile_models'][i] = GradientBoostingRegressor(n_estimators=600, loss='quantile', alpha=quantile)
                        self.targets[target_col_name]['quantile_models'][i].fit(X, Y)

            else:
                self.targets[target_col_name]['model'] = None

    def predict(self, when_data_source, include_extra_data=False):
        X = []
        for row in when_data_source:
            X.append(self._row_to_ndarray(row))

        predictions = {}

        for target_col_name in self.targets:

            if self.targets[target_col_name]['model'] is None:
                predictions[target_col_name] = None
            else:
                predictions[target_col_name] = {}
                predictions[target_col_name]['predictions'] = [x for x in self.targets[target_col_name]['model'].predict(X)]

                try:
                    predictions[target_col_name]['selfaware_confidences'] = [max(x) for x in self.targets[target_col_name]['model'].predict_proba(X)]
                except Exception as e:
                    pass

                if 'quantile_models' in self.targets[target_col_name]:
                    lower_quantiles = self.targets[target_col_name]['quantile_models'][0].predict(X)
                    upper_quantiles = self.targets[target_col_name]['quantile_models'][1].predict(X)

                    predictions[target_col_name]['confidence_range'] = [[lower_quantiles[i],upper_quantiles[i]] for i in range(len(lower_quantiles))]
                    predictions[target_col_name]['quantile_confidences'] = [self.quantiles[1] - self.quantiles[0] for i in range(len(lower_quantiles))]

        return predictions
