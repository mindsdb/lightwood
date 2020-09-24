import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, SGDRegressor, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, r2_score, balanced_accuracy_score

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
        self.targets = None
        self.transformer = None
        self.encoders = None

    def fit(self, train_ds, test_ds=None):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """
        self.fit_data_source(train_ds)

        self.targets = {}

        if test_ds is None:
            test_ds = train_ds.subset(0.1)
        else:
            self.fit_data_source(test_ds)
            assert test_ds.transformer is train_ds.transformer
            assert test_ds.encoders is train_ds.encoders

        self.transformer = train_ds.transformer
        self.encoders = train_ds.encoders
        
        for output_feature in train_ds.output_features:
            self.targets[output_feature['name']] = {
                'type': output_feature['type']
            }
            if 'weights' in output_feature:
                self.targets[output_feature['name']]['weights'] = output_feature['weights']
            else:
                self.targets[output_feature['name']]['weights'] = None

        X_train = []
        for row in train_ds:
            print('train:', row)
            X_train.append(np.array(row[0]))

        X_test = []
        for row in test_ds:
            print('test:', row)
            X_test.append(np.array(row[0]))

        assert len(X_train) > 0
        assert len(X_test) > 0

        for target_col_name in self.targets:
            Y_train = train_ds.get_column_original_data(target_col_name)
            Y_test = test_ds.get_column_original_data(target_col_name)

            if self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.CATEGORICAL:
                weight_map = self.targets[target_col_name]['weights']
                if weight_map is None:
                    sample_weight = [1] * len(Y_train)
                else:
                    sample_weight = [weight_map[val] for val in Y_train]

                model_classes_and_accuracies = []
                for model_class, model_kwargs in CLASSIFICATION_MODELS:
                    model = model_class(**model_kwargs)
                    model.fit(X_train, Y_train, sample_weight=sample_weight)
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

            elif self.targets[target_col_name]['type'] == COLUMN_DATA_TYPES.NUMERIC:
                model_classes_and_accuracies = []
                for model_class, model_kwargs in REGRESSION_MODELS:
                    model = model_class(**model_kwargs)
                    model.fit(X_train, Y_train)
                    accuracy = 0#r2_score(Y_test, model.predict(X_test))

                    model_classes_and_accuracies.append((
                        (model_class, model_kwargs),
                        accuracy
                    ))
                
                (best_model_class, best_model_kwargs), _ = max(
                    model_classes_and_accuracies,
                    key=lambda x: x[-1]
                )

                self.targets[target_col_name]['model'] = best_model_class(**best_model_kwargs)

                # TODO
                # if self.quantiles is not None:
                #     self.targets[target_col_name]['quantile_models'] = {}
                #     for i, quantile in enumerate(self.quantiles):
                #         self.targets[target_col_name]['quantile_models'][i] = best_model_class(
                #             best_model_kwargs,
                #             loss='quantile',
                #             alpha=quantile
                #         )
                #         self.targets[target_col_name]['quantile_models'][i].fit(X, Y)

            else:
                self.targets[target_col_name]['model'] = None
        
        # Fit best model of each column on [train_ds + test_ds]
        for target_col_name in self.targets:
            Y_train = train_ds.get_column_original_data(target_col_name)
            Y_test = test_ds.get_column_original_data(target_col_name)
            X, Y = (X_train + X_test), (Y_train + Y_test)

            if self.targets[target_col_name]['model'] is not None:
                self.targets[target_col_name]['model'].fit(X, Y)

            if 'quantile_models' in self.targets[target_col_name]:
                for model in self.targets[target_col_name]['quantile_models']:
                    model.fit(X, Y)
    
    def predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """
        assert self.transformer is not None and self.encoders is not None, 'first fit the mixer'
        when_data_source.transformer = self.transformer
        when_data_source.encoders = self.encoders
        _, _ = when_data_source[0]

        X = []
        for row in when_data_source:
            X.append(np.array(row[0]))
        
        predictions = {}

        for target_col_name in self.targets:
            if self.targets[target_col_name]['model'] is None:
                predictions[target_col_name] = None
            else:
                predictions[target_col_name] = {}
                predictions[target_col_name]['predictions'] = list(self.targets[target_col_name]['model'].predict(X))

                try:
                    predictions[target_col_name]['selfaware_confidences'] = [max(x) for x in self.targets[target_col_name]['model'].predict_proba(X)]
                except Exception as e:
                    pass

                # if 'quantile_models' in self.targets[target_col_name]:
                #     lower_quantiles = self.targets[target_col_name]['quantile_models'][0].predict(X)
                #     upper_quantiles = self.targets[target_col_name]['quantile_models'][1].predict(X)

                #     predictions[target_col_name]['confidence_range'] = [[lower_quantiles[i],upper_quantiles[i]] for i in range(len(lower_quantiles))]
                #     predictions[target_col_name]['quantile_confidences'] = [self.quantiles[1] - self.quantiles[0] for i in range(len(lower_quantiles))]

        return predictions
