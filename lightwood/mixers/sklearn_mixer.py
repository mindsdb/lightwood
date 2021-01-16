import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier

from lightwood.logger import log
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.mixers import BaseMixer

CLASSIFICATION_MODELS = [
    (SGDClassifier, {}),
    (SVC, {}),
    (GaussianNB, {}),
]

REGRESSION_MODELS = [
    (LinearRegression, {}),
    (SGDRegressor, {}),
]


class SklearnMixer(BaseMixer):
    def __init__(self):
        super().__init__()
        self.binarizers = {}

    def _fit(self, train_ds, test_ds):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """
        X_train = []
        for row in train_ds:
            X_train.append(np.array(row[0]))

        X_test = []
        for row in test_ds:
            X_test.append(np.array(row[0]))

        for target_col_name in self.targets:
            Y_train = train_ds.get_column_original_data(target_col_name)
            Y_test = test_ds.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1] * len(Y_train)
                else:
                    sample_weight = [weight_map[str(val)] for val in Y_train]

                model_classes_and_accuracies = []
                for model_class, model_kwargs in CLASSIFICATION_MODELS:
                    model = model_class(**model_kwargs)
                    model.fit(
                        X_train,
                        Y_train,
                        sample_weight=sample_weight
                    )

                    accuracy = balanced_accuracy_score(
                        Y_test,
                        model.predict(X_test)
                    )

                    model_classes_and_accuracies.append((
                        (model_class, model_kwargs),
                        accuracy
                    ))

                (best_model_class, best_model_kwargs), _ = max(
                    model_classes_and_accuracies,
                    key=lambda x: x[-1]
                )

                self.targets[target_col_name]['model'] = best_model_class(**best_model_kwargs)

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1] * len(Y_train)
                else:
                    sample_weight = [weight_map[val] for val in Y_train]

                model_classes_and_accuracies = []
                for model_class, model_kwargs in CLASSIFICATION_MODELS:
                    self.binarizers[target_col_name] = MultiLabelBinarizer(sparse_output=True)
                    model = MultiOutputClassifier(
                        model_class(**model_kwargs)
                    )
                    model.fit(
                        X_train,
                        self.binarizers[target_col_name].fit_transform(Y_train).toarray(),
                        sample_weight=sample_weight
                    )

                    accuracy = accuracy_score(
                        self.binarizers[target_col_name].transform(Y_test),
                        model.predict(X_test)
                    )

                    model_classes_and_accuracies.append((
                        (model_class, model_kwargs),
                        accuracy
                    ))

                (best_model_class, best_model_kwargs), _ = max(
                    model_classes_and_accuracies,
                    key=lambda x: x[-1]
                )

                self.targets[target_col_name]['model'] = MultiOutputClassifier(
                    best_model_class(**best_model_kwargs)
                )

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.NUMERIC:
                model_classes_and_accuracies = []
                for model_class, model_kwargs in REGRESSION_MODELS:
                    model = model_class(**model_kwargs)
                    model.fit(X_train, Y_train)

                    accuracy = r2_score(
                        Y_test,
                        model.predict(X_test)
                    )

                    model_classes_and_accuracies.append((
                        (model_class, model_kwargs),
                        accuracy
                    ))

                (best_model_class, best_model_kwargs), best_accuracy = max(
                    model_classes_and_accuracies,
                    key=lambda x: x[-1]
                )

                self.targets[target_col_name]['model'] = best_model_class(**best_model_kwargs)
            else:
                self.targets[target_col_name]['model'] = None

        # Fit best model of each column on [train_ds + test_ds]
        for target_col_name in self.targets:
            Y_train = train_ds.get_column_original_data(target_col_name)
            Y_test = test_ds.get_column_original_data(target_col_name)
            X, Y = (X_train + X_test), (Y_train + Y_test)

            if self.targets[target_col_name]['model'] is not None:
                if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                    self.targets[target_col_name]['model'].fit(
                        X,
                        self.binarizers[target_col_name].fit_transform(Y).toarray()
                    )
                else:
                    self.targets[target_col_name]['model'].fit(X, Y)

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
                    predictions[target_col_name]['predictions'] = list(
                        self.targets[target_col_name]['model'].predict(X)
                    )

                    # add distribution belief if the flag was set in the target encoder
                    if getattr(self.encoders[target_col_name], 'predict_proba', False):
                        predictions[target_col_name]['class_distribution'] = self.targets[target_col_name]['model'].decision_function(X)
                        predictions[target_col_name]['class_labels'] = {i:cls for i, cls in enumerate(self.targets[target_col_name]['model'].classes_)}

                try:
                    predictions[target_col_name]['selfaware_confidences'] = [max(x) for x in self.targets[target_col_name]['model'].predict_proba(X)]
                except Exception:
                    pass

        return predictions
