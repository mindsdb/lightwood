import logging
import warnings

import torch
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from lightwood.mixers.sk_learn.sk_learn_helper import SkLearnMixerHelper


class SkLearnMixer(SkLearnMixerHelper):

    def __init__(self, score_threshold=0.5,
                 classifier_class=MultiOutputClassifier, regression_class=MultiOutputRegressor):
        """
        :param input_column_names: is a list [col_name1, col_name2]
        :param output_column_names: is a list [col_name1, col_name2]
        :param score_threshold: score to be considered for each column
        :param classifier_class: model name for classification
        :param regression_class: model name for Regression
        """
        self.input_column_names = None
        self.output_column_names = None
        self.feature_columns = {}  # the columns that are actually used in the fit and predict
        self.output_encoders = {}
        self.score_threshold = score_threshold
        self.classifier_class = classifier_class
        self.regression_class = regression_class
        self.model = {}
        self.feature_models = {}
        self.feature_importance = {}
        self.encoders = None

    def fit(self, data_source):
        """
        :param data_source: is a DataSource object
        :return model: fitted model
        """
        logging.info('Model training started')
        for column in self.output_column_names:
            model_class = self._determine_model_class(column, data_source)
            output_encoded_column = self._output_encoded_columns(column, data_source)

            useful_input_encoded_features, self.feature_columns[column] = self._extract_features(
                data_source, model_class, output_encoded_column)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model[column] = model_class.fit(useful_input_encoded_features, output_encoded_column)

            model_score = self.model[column].score(useful_input_encoded_features, output_encoded_column)

        logging.info('Model training completed with score:{}'.format(model_score))
        self.encoders = data_source.encoders
        return self.model

    def overall_certainty(self):
        return -1

    def predict(self, when_data_source, output_column_names=None):
        """
        :param when_data_source: is a DataSource object
        :param output_column_names: is a DataSource object
        :return predictions: numpy.ndarray predicted encoded values
        """
        logging.info('Model predictions starting')
        model = self.model
        when_data_source.encoders = self.encoders
        output_column_names = self.output_column_names if output_column_names is None else output_column_names
        predictions = dict()
        for output_column in output_column_names:
            input_encoded = self._input_encoded_columns(output_column, when_data_source)
            encoded_predictions = model.get(output_column).predict(input_encoded)

            decoded_predictions = self._decoded_data([output_column], when_data_source,
                                                     torch.from_numpy(encoded_predictions))
            predictions[output_column] = {'Encoded Predictions': encoded_predictions,
                                          'predictions': decoded_predictions}

        logging.info('Model predictions and decoding completed')
        return predictions

    def error(self, ds):
        """
        :param ds: is a DataSource Object
        :return: error :Dictionary: error of actual vs predicted encoded values
        """
        error = {}
        predictions = self.predict(ds)
        for output_column in self.output_column_names:
            error[output_column] = mean_squared_error(ds.encoded_cache[output_column].numpy(),
                                                      predictions[output_column]['Encoded Predictions'])
        return error

    def iter_fit(self, ds):
        """
        :param ds:  is a DataSource object
        :return error : Dictionary: error of actual vs predicted encoded values
        """
        self.input_column_names = self.input_column_names \
            if self.input_column_names is not None else ds.get_feature_names('input_features')
        self.output_column_names = self.output_column_names \
            if self.output_column_names is not None else ds.get_feature_names('output_features')
        self.encoders = ds.encoders
        for i in range(1):
            self.fit(ds)
            yield self.error(ds)


if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource

    ###############
    # GENERATE DATA
    ###########################
    # Test Case 1             #
    # For Classification      #
    ###########################
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
    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]
    data['z'] = ['low' if i < 50 else 'high' for i in nums]
    data_frame = pandas.DataFrame(data)
    print(data_frame)

    ds = DataSource(data_frame, config)
    input_ds_for_prediction = DataSource(data_frame[['x', 'y']], config)

    mixer = SkLearnMixer(input_column_names=['x', 'y'], output_column_names=['z'])
    for i in mixer.iter_fit(ds):
        print('training')

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(input_ds_for_prediction, ['z'])
    print(predictions)

    #####################################
    # For Regression                    #
    # Test Case: 2                      #
    #####################################
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
    print(data_frame)

    ds = DataSource(data_frame, config)
    input_ds_for_prediction = DataSource(data_frame[['x', 'y']], config)

    mixer = SkLearnMixer(input_column_names=['x', 'y'], output_column_names=['z'])

    for i in mixer.iter_fit(ds):
        print('training')

    predictions = mixer.predict(input_ds_for_prediction, ['z'])
    print(predictions)

    #########################################
    # Multiple Target variables             #
    # Test Case 3                           #
    #########################################
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
                'name': 'z1',
                'type': 'categorical',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            },
            {
                'name': 'z2',
                'type': 'numeric',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]
    data['z1'] = ['low' if i < 50 else 'high' for i in nums]
    data['z2'] = [i + 0.5 for i in range(10)]
    data_frame = pandas.DataFrame(data)
    print(data_frame)

    ds = DataSource(data_frame, config)
    input_ds_for_prediction = DataSource(data_frame[['x', 'y']], config)

    mixer = SkLearnMixer(input_column_names=['x', 'y'], output_column_names=['z1', 'z2'])
    for i in mixer.iter_fit(ds):
        print('training')
        print(i)
    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(input_ds_for_prediction, ['z1', 'z2'])
    print(predictions)
