import logging
import warnings

import numpy as np
import torch
from sklearn import svm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class SKLearnMixer:

    def __init__(self, input_column_names, output_column_names, score_threshold=0.5,
                 classifier_class=MultiOutputClassifier, regression_class=svm):
        """
        :param input_column_names: is a list [col_name1, col_name2]
        :param output_column_names: is a list [col_name1, col_name2]
        :param score_threshold: score to be considered for each column
        :param classifier_class: model name for classification
        :param regression_class: model name for Regression
        """
        self.input_column_names = input_column_names
        self.output_column_names = output_column_names
        self.feature_columns = []  # the columns that are actually used in the fit and predict
        self.output_encoders = {}
        self.score_threshold = score_threshold
        self.classifier_class = classifier_class
        self.regression_class = regression_class
        self.model = None
        self.feature_models = {}
        self.feature_importance = {}

    def fit(self, data_source):
        """
        :param data_source: is a DataSource object
        :return model: fitted model
        """
        logging.info('Model training started')

        # ToDo: Should be able to handle multiple target variables
        model_class = [self._model_class(column, data_source) for column in self.output_column_names][0]

        output_encoded_column = self._output_encoded_columns(data_source)

        useful_input_encoded_features = self._extract_features(data_source, model_class, output_encoded_column)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = model_class.fit(useful_input_encoded_features, output_encoded_column)

        model_score = self.model.score(useful_input_encoded_features, output_encoded_column)
        logging.info('Model training completed with score:{}'.format(model_score))
        return self.model

    def predict(self, when_data_source):
        """
        :param when_data_source: is a DataSource object
        :return predictions: numpy.ndarray predicted encoded values
        """
        logging.info('Model predictions starting')
        input_encoded = self._input_encoded_columns(when_data_source)

        encoded_predictions = self.model.predict(input_encoded)

        decoded_predictions = self._decoded_data(self.output_column_names, when_data_source,
                                                 torch.from_numpy(encoded_predictions))
        logging.info('Model predictions and decoding completed')
        return {'Encoded Predictions': encoded_predictions,
                'Actual Predictions ': decoded_predictions}

    def _input_encoded_columns(self, when_data_source):
        """
        :param when_data_source: is a DataSource object
        :return: numpy.nd array input encoded values
        """
        input_encoded = None
        for column in self.feature_columns:
            if input_encoded is None:
                input_encoded = self._encoded_data([column], when_data_source)
            else:
                np.append(input_encoded, self._encoded_data([column], when_data_source))
        return StandardScaler().fit_transform(input_encoded)

    def _output_encoded_columns(self, data_source):
        """
        :param data_source: is a DataSource object
        :return: numpy.nd array output encoded values
        """
        output_encoded_column = self._encoded_data(self.output_column_names, data_source)
        self.output_encoders = data_source.encoders
        return output_encoded_column

    def _extract_features(self, data_source, model_class, output_encoded_column):
        """
        :param data_source: is a DataSource object
        :param model_class: type of model to be fitted
        :param output_encoded_column: target variable encoded values
        :return: numpy.nd array: important input encoded columns
        """
        input_encoded_columns = None
        for column in self.input_column_names:
            input_encoded_column = self._encoded_data([column], data_source)
            input_encoded_column = StandardScaler().fit_transform(input_encoded_column)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_class.fit(input_encoded_column, output_encoded_column)
            score = model.score(input_encoded_column, output_encoded_column)
            self.feature_models[column] = model
            self.feature_importance[column] = score

            if score > self.score_threshold:
                self.feature_columns.append(column)
                if input_encoded_columns is None:
                    input_encoded_columns = input_encoded_column
                else:
                    np.append(input_encoded_columns, input_encoded_column, axis=1)
        return StandardScaler().fit_transform(input_encoded_columns)

    def _encoded_data(self, features, data_source):
        """
        :param features: list of column names
        :param data_source: input data
        :return encoded_data: numpy.nd array encoded values
        """
        for cnt, column in enumerate(features):
            if cnt == 0:
                encoded_data = data_source.get_encoded_column_data(column).numpy()
            else:
                np.append(encoded_data, data_source.get_encoded_column_data(column).numpy(), axis=1)
        return encoded_data

    def _decoded_data(self, features, data_source, encoded_data):
        """
        :param features: list : columns to be decoded
        :param data_source: is a DataSource object
        :param encoded_data: encoded data
        :return:  decoded data
        """
        data_source.encoders = self.output_encoders
        for column in features:
            encoders = self.output_encoders.get(column, None)
            if encoders is None:
                decoded_data = data_source.get_decoded_column_data(column, encoded_data)
            else:
                decoded_data = encoders.decode(encoded_data)
        return decoded_data

    def _model_class(self, column, data_source):
        """
        :param column: name of the column
        :param data_source: is a DataSource object
        :return: model: Model to be considered for fitting data
        """
        data_type = None
        for feature in data_source.configuration['output_features']:
            if feature['name'] == column:
                data_type = feature['type']
                break

        if data_type is not None:
            return self._get_model(data_type)

        return None

    def _get_model(self, column_type):
        models = {
            'categorical': self.classifier_class(KNeighborsClassifier(3), n_jobs=-1),
            'numeric': MultiOutputRegressor(svm.SVR())
        }
        return models.get(column_type, None)


if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource

    ###############
    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    ##For Classification
    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i < 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    # print(data_frame)

    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = SKLearnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(predict_input_ds)
    print(predictions)

    ##For Regression

    # GENERATE DATA
    ###############

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            },
            {
                'name': 'y',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = [i + 0.5 for i in range(10)]

    data_frame = pandas.DataFrame(data)
    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = SKLearnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(predict_input_ds)
    print(predictions)
