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

                    # add distribution belief if the flag was set in the target encoder
                    if getattr(self.encoders[target_col_name], 'predict_proba', False):
                        predictions[target_col_name]['class_distribution'] = self.targets[target_col_name]['model'].predict_proba(X)
                        predictions[target_col_name]['class_labels'] = {i:cls for i, cls in enumerate(self.targets[target_col_name]['model'].classes_)}

                try:
                    predictions[target_col_name]['selfaware_confidences'] = [max(x) for x in self.targets[target_col_name]['model'].predict_proba(X)]
                except Exception:
                    pass
        return predictions
