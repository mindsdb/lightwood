import logging
import traceback
import time

import dill
import pandas
import numpy as np
import torch

from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.config.config import CONFIG
from lightwood.mixers.nn.nn import NnMixer
from sklearn.metrics import accuracy_score, r2_score, f1_score
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices

class Predictor:

    def __init__(self, config=None, output=None, load_from_path=None):
        """
        Start a predictor pass the

        :param config: a predictor definition object (can be a dictionary or a PredictorDefinition object)
        :param output: the columns you want to predict, ludwig will try to generate a config
        :param load_from_path: The path to load the predictor from
        :type config: dictionary
        """
        try:
            from lightwood.mixers.boost.boost import BoostMixer
            self.has_boosting_mixer = True
        except Exception as e:
            self.has_boosting_mixer = False
            logging.info(f'Boosting mixer can\'t be loaded due to error: {e} !')
            print((f'Boosting mixer can\'t be loaded due to error: {e} !'))

        if load_from_path is not None:
            pickle_in = open(load_from_path, "rb")
            self_dict = dill.load(pickle_in)
            pickle_in.close()
            self.__dict__ = self_dict
            self.convert_to_device()
            return

        if output is None and config is None:
            raise ValueError('You must give one argument to the Predictor constructor')
        try:
            if config is not None and output is None:
                self.config = predictor_config_schema.validate(config)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        # this is if we need to automatically generate a configuration variable
        self._generate_config = True if output is not None or self.config is None else False

        self._output_columns = output
        self._input_columns = None
        self.train_accuracy = None

        self._mixer = None
        self._helper_mixers = None

    @staticmethod
    def evaluate_mixer(config, mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters,
                       max_training_time=None, max_epochs=None):
        started_evaluation_at = int(time.time())
        lowest_error = 10000
        mixer = mixer_class(dynamic_parameters, config)

        if max_training_time is None and max_epochs is None:
            err = "Please provide either `max_training_time` or `max_epochs` when calling `evaluate_mixer`"
            logging.error(err)
            raise Exception(err)

        lowest_error_epoch = 0
        for epoch, training_error in enumerate(mixer.iter_fit(from_data_ds)):
            error = mixer.error(test_data_ds)

            if lowest_error > error:
                lowest_error = error
                lowest_error_epoch = epoch

            if max(lowest_error_epoch * 1.4, 10) < epoch:
                return lowest_error

            if max_epochs is not None and epoch >= max_epochs:
                return lowest_error

            if max_training_time is not None and started_evaluation_at < (int(time.time()) - max_training_time):
                return lowest_error

    def convert_to_device(self, device_str=None):
        if device_str is not None:
            device = torch.device(device_str)
            available_devices = 1
            if device_str == 'cuda':
                available_devices = torch.cuda.device_count()
        else:
            device, available_devices = get_devices()

        self._mixer.to(device, available_devices)
        for e in self._mixer.encoders:
            self._mixer.encoders[e].to(device, available_devices)

    def train_helper_mixers(self, train_ds, test_ds, quantiles):
        from lightwood.mixers.boost.boost import BoostMixer

        boost_mixer = BoostMixer(quantiles=quantiles)
        boost_mixer.fit(train_ds=train_ds)

        # @TODO: IF we add more mixers in the future, add the best on for each column to this map !
        best_mixer_map = {}
        predictions = boost_mixer.predict(test_ds)

        for output_column in self._output_columns:
            model = boost_mixer.targets[output_column]['model']
            if model is None:
                continue

            real = list(map(str,test_ds.get_column_original_data(output_column)))
            predicted =  predictions[output_column]['predictions']

            weight_map = None
            if 'weights' in test_ds.get_column_config(output_column):
                weight_map = train_ds.get_column_config(output_column)['weights']

            accuracy = self.apply_accuracy_function(train_ds.get_column_config(output_column)['type'], real, predicted, weight_map)
            best_mixer_map[output_column] = {
                'model': boost_mixer
                ,'accuracy': accuracy['value']
            }
        return best_mixer_map


    def learn(self, from_data, test_data=None, callback_on_iter=None, eval_every_x_epochs=20, stop_training_after_seconds=None, stop_model_building_after_seconds=None):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data: (Pandas DataFrame) The data to learn from
        :param test_data: (Pandas DataFrame) The data to test accuracy and learn_error from
        :param callback_on_iter: This is function that can be called on every X evaluation cycle
        :param eval_every_x_epochs: This is every how many epochs we want to calculate the test error and accuracy

        :return: None
        """

        # This is a helper function that will help us auto-determine roughly what data types are in each column
        # NOTE: That this assumes the data is clean and will only return types for 'CATEGORICAL', 'NUMERIC' and 'TEXT'
        def type_map(col_name):
            col_pd_type = from_data[col_name].dtype
            col_pd_type = str(col_pd_type)

            if col_pd_type in ['int64', 'float64', 'timedelta']:
                return COLUMN_DATA_TYPES.NUMERIC
            elif col_pd_type in ['bool', 'category']:
                return COLUMN_DATA_TYPES.CATEGORICAL
            else:
                # if the number of uniques is elss than 100 or less,
                # than 10% of the total number of rows then keep it as categorical
                unique = from_data[col_name].nunique()
                if unique < 100 or unique < len(from_data[col_name]) / 10:
                    return COLUMN_DATA_TYPES.CATEGORICAL
                # else assume its text
                return COLUMN_DATA_TYPES.TEXT

        # generate the configuration and set the order for the input and output columns
        if self._generate_config is True:
            self._input_columns = [col for col in from_data if col not in self._output_columns]
            self.config = {
                'input_features': [{'name': col, 'type': type_map(col)} for col in self._input_columns],
                'output_features': [{'name': col, 'type': type_map(col)} for col in self._output_columns]
            }
            self.config = predictor_config_schema.validate(self.config)
            logging.info('Automatically generated a configuration')
            logging.info(self.config)
        else:
            self._output_columns = [col['name'] for col in self.config['output_features']]
            self._input_columns = [col['name'] for col in self.config['input_features']]

        if stop_training_after_seconds is None:
            stop_training_after_seconds = round(from_data.shape[0] * from_data.shape[1] / 5)

        if stop_model_building_after_seconds is None:
            stop_model_building_after_seconds = stop_training_after_seconds * 3

        from_data_ds = DataSource(from_data, self.config)

        if test_data is not None:
            test_data_ds = DataSource(test_data, self.config)
        else:
            test_data_ds = from_data_ds.extractRandomSubset(0.1)

        from_data_ds.training = True

        mixer_class = NnMixer
        mixer_params = {}

        if 'mixer' in self.config:
            if 'class' in self.config['mixer']:
                mixer_class = self.config['mixer']['class']
            if 'attrs' in self.config['mixer']:
                mixer_params = self.config['mixer']['attrs']

        # Initialize data sources
        if len(from_data_ds) > 100:
            nr_subsets = 3
        else:
            # Don't use k-fold cross validation for very small input sizes
            nr_subsets = 1

        from_data_ds.prepare_encoders()
        from_data_ds.create_subsets(nr_subsets)
        try:
            mixer_class({}).fit_data_source(from_data_ds)
        except Exception as e:
            # Not all mixers might require this
            # print(e)
            pass

        input_size = len(from_data_ds[0][0])
        training_data_length = len(from_data_ds)

        test_data_ds.transformer = from_data_ds.transformer
        test_data_ds.encoders = from_data_ds.encoders
        test_data_ds.output_weights = from_data_ds.output_weights
        test_data_ds.create_subsets(nr_subsets)

        if 'optimizer' in self.config:
            optimizer = self.config['optimizer']()

            while True:
                training_time_per_iteration = stop_model_building_after_seconds / optimizer.total_trials

                # Some heuristics...
                if training_time_per_iteration > input_size:
                    if training_time_per_iteration > min((training_data_length / (4 * input_size)), 16 * input_size):
                        break

                optimizer.total_trials = optimizer.total_trials - 1
                if optimizer.total_trials < 8:
                    optimizer.total_trials = 8
                    break

            training_time_per_iteration = stop_model_building_after_seconds / optimizer.total_trials

            best_parameters = optimizer.evaluate(lambda dynamic_parameters: Predictor.evaluate_mixer(self.config, mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters, max_training_time=training_time_per_iteration, max_epochs=None))

            logging.info('Using hyperparameter set: ', best_parameters)
        else:
            best_parameters = {}

        self._mixer = mixer_class(best_parameters, self.config)

        for param in mixer_params:
            if hasattr(self._mixer, param):
                setattr(self._mixer, param, mixer_params[param])
            else:
                logging.warning(
                    'trying to set mixer param {param} but mixerclass {mixerclass} does not have such parameter'.format
                    (param=param, mixerclass=str(type(self._mixer)))
                )

        def callback_on_iter_w_acc(epoch, training_error, test_error, delta_mean):
            if callback_on_iter is not None:
                callback_on_iter(epoch, training_error, test_error, delta_mean, self.calculate_accuracy(test_data_ds))

        self._mixer.fit(train_ds=from_data_ds ,test_ds=test_data_ds, callback=callback_on_iter_w_acc, stop_training_after_seconds=stop_training_after_seconds, eval_every_x_epochs=eval_every_x_epochs)
        self.train_accuracy = self.calculate_accuracy(test_data_ds)

        # Train some alternative mixers
        if CONFIG.HELPER_MIXERS and self.has_boosting_mixer and (CONFIG.FORCE_HELPER_MIXERS or len(from_data_ds) < 12 * pow(10,3)):
            try:
                self._helper_mixers = self.train_helper_mixers(from_data_ds, test_data_ds, self._mixer.quantiles[self._mixer.quantiles_pair[0]+1:self._mixer.quantiles_pair[1]+1])
            except Exception as e:
                logging.warning(f'Failed to train helper mixers with error: {e}')

        return self

    def predict(self, when_data=None, when=None):
        """
        Predict given when conditions
        :param when_data: a dataframe
        :param when: a dictionary
        :return: a complete dataframe
        """
        if when is not None:
            when_dict = {key: [when[key]] for key in when}
            when_data = pandas.DataFrame(when_dict)

        when_data_ds = DataSource(when_data, self.config)
        when_data_ds.encoders = self._mixer.encoders

        main_mixer_predictions = self._mixer.predict(when_data_ds)

        if CONFIG.HELPER_MIXERS and self.has_boosting_mixer:
            for output_column in main_mixer_predictions:
                if self._helper_mixers is not None and output_column in self._helper_mixers:
                    if (self._helper_mixers[output_column]['accuracy'] > 1.00 * self.train_accuracy[output_column]['value']) or CONFIG.FORCE_HELPER_MIXERS:
                        helper_mixer_predictions = self._helper_mixers[output_column]['model'].predict(when_data_ds, [output_column])

                        main_mixer_predictions[output_column] = helper_mixer_predictions[output_column]

        return main_mixer_predictions

    @staticmethod
    def apply_accuracy_function(col_type, real, predicted, weight_map=None, encoder=None):
        if col_type == COLUMN_DATA_TYPES.CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in real]
            else:
                sample_weight = []
                for val in real:
                    sample_weight.append(weight_map[val])

            accuracy = {
                'function': 'accuracy_score',
                'value': accuracy_score(real, predicted, sample_weight=sample_weight)
            }
        elif col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in real]
            else:
                sample_weight = []
                for val in real:
                    sample_weight.append(weight_map[val])

            encoded_real = encoder.encode(real)
            encoded_predicted = encoder.encode(predicted)

            accuracy = {
                'function': 'f1_score',
                'value': f1_score(encoded_real, encoded_predicted, average='weighted', sample_weight=sample_weight)
            }
        else:
            real_fixed = []
            predicted_fixed = []
            for val in real:
                try:
                    real_fixed.append(float(val))
                except:
                    real_fixed.append(0)

            for val in predicted:
                try:
                    predicted_fixed.append(float(val))
                except:
                    predicted_fixed.append(0)

            accuracy = {
                'function': 'r2_score',
                'value': r2_score(real_fixed, predicted_fixed)
            }
        return accuracy

    def calculate_accuracy(self, from_data):
        """
        calculates the accuracy of the model
        :param from_data:a dataframe
        :return accuracies: dictionaries of accuracies
        """

        if self._mixer is None:
            logging.error("Please train the model before calculating accuracy")
            return
        ds = from_data if isinstance(from_data, DataSource) else DataSource(from_data, self.config)
        predictions = self._mixer.predict(ds, include_extra_data=True)
        accuracies = {}

        for output_column in self._output_columns:

            col_type = ds.get_column_config(output_column)['type']

            if col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                real = list(map(tuple, ds.get_column_original_data(output_column)))
                predicted = list(map(tuple, predictions[output_column]['predictions']))
            else:
                real = list(map(str,ds.get_column_original_data(output_column)))
                predicted = list(map(str,predictions[output_column]['predictions']))

            weight_map = None
            if 'weights' in ds.get_column_config(output_column):
                weight_map = ds.get_column_config(output_column)['weights']

            accuracy = self.apply_accuracy_function(ds.get_column_config(output_column)['type'],
                                                    real,
                                                    predicted,
                                                    weight_map=weight_map,
                                                    encoder=ds.encoders[output_column])

            if ds.get_column_config(output_column)['type'] == COLUMN_DATA_TYPES.NUMERIC:
                ds.encoders[output_column].decode_log = True
                predicted = ds.get_decoded_column_data(output_column, predictions[output_column]['encoded_predictions'])

                alternative_accuracy = self.apply_accuracy_function(ds.get_column_config(output_column)['type'], real, predicted,weight_map=weight_map)

                if alternative_accuracy['value'] > accuracy['value']:
                    accuracy = alternative_accuracy
                else:
                    ds.encoders[output_column].decode_log = False

            accuracies[output_column] = accuracy

        return accuracies

    def save(self, path_to):
        """
        save trained model to a file
        :param path_to: full path of file, where we store results
        :return:
        """
        f = open(path_to, 'wb')

        # Null out certain object we don't want to store
        if hasattr(self._mixer, '_nonpersistent'):
            self._mixer._nonpersistent = {}

        # Dump everything relevant to cpu before saving
        self.convert_to_device("cpu")
        dill.dump(self.__dict__, f)
        self.convert_to_device()
        f.close()
