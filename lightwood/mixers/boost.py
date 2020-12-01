import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import BaseMixer


class BoostMixer(BaseMixer):
    def __init__(self):
        super().__init__()
        self.binarizers = {}

    def _fit(self, train_ds, test_ds=None):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """
        # If test data is provided, use it for trainig
        if test_ds is not None:
            train_ds.extend(test_ds.data_frame)

        X = []

        for row in train_ds:
            X.append(np.array(row[0]))

        for target_col_name in self.targets:
            Y = train_ds.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1] * len(Y)
                else:
                    sample_weight = [weight_map[str(val)] for val in Y]

                self.targets[target_col_name]['model'] = GradientBoostingClassifier(n_estimators=600)
                self.targets[target_col_name]['model'].fit(X, Y, sample_weight=sample_weight)

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1] * len(Y)
                else:
                    sample_weight = [weight_map[val] for val in Y]

                self.binarizers[target_col_name] = MultiLabelBinarizer(sparse_output=True)
                self.targets[target_col_name]['model'] = MultiOutputClassifier(
                    GradientBoostingClassifier(n_estimators=600)
                )
                self.targets[target_col_name]['model'].fit(
                    X,
                    self.binarizers[target_col_name].fit_transform(Y).toarray()
                )

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

    def _predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """
        X = []
        for row in when_data_source:
            X.append(np.array(row[0]))
        
        predictions = {}

        for target_col_name in self.targets:
            if self.targets[target_col_name]['model'] is None:
                predictions[target_col_name] = None
            else:
                predictions[target_col_name] = {}

                if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                    predictions[target_col_name]['predictions'] = list(
                        self.binarizers[target_col_name].inverse_transform(
                            self.targets[target_col_name]['model'].predict(X)
                        )
                    )
                else:
                    predictions[target_col_name]['predictions'] = list(self.targets[target_col_name]['model'].predict(X))

                try:
                    predictions[target_col_name]['selfaware_confidences'] = [max(x) for x in self.targets[target_col_name]['model'].predict_proba(X)]
                except Exception:
                    pass

                if 'quantile_models' in self.targets[target_col_name]:
                    lower_quantiles = self.targets[target_col_name]['quantile_models'][0].predict(X)
                    upper_quantiles = self.targets[target_col_name]['quantile_models'][1].predict(X)

                    predictions[target_col_name]['confidence_range'] = [[lower_quantiles[i],upper_quantiles[i]] for i in range(len(lower_quantiles))]
                    predictions[target_col_name]['quantile_confidences'] = [self.quantiles[1] - self.quantiles[0] for i in range(len(lower_quantiles))]

        return predictions
