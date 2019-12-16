import logging
import traceback
import time
import math
import copy
import sys

import dill
import pandas
import numpy as np
import torch

from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.config.config import CONFIG
from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
from lightwood.mixers.nn.nn import NnMixer
from sklearn.metrics import accuracy_score, r2_score
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


class Predictor:

    def __init__(self, config=None, output=None, load_from_path=None):
        """
        Start a predictor pass the

        :param config: a predictor definition object (can be a dictionary or a PredictorDefinition object)
        :param output: the columns you want to predict, ludwig will try to generate a config
        :param load_from_path: The path to load the predictor from
        :type config: dictionary
        """

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
                predictor_config_schema.validate(config)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        self.config = config

        self._generate_config = True if output is not None else False # this is if we need to automatically generate a configuration variable

        self._output_columns = output
        self._input_columns = None

        self._encoders = None
        self._mixer = None
        self._mixers = {}
        self._stop_training_flag = False

        self.train_accuracy = None
        self.overall_certainty = None

    @staticmethod
    def evaluate_mixer(mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters, is_categorical_output, max_training_time=None, max_epochs=None):
        started_evaluation_at = int(time.time())
        lowest_error = 1
        mixer = mixer_class(dynamic_parameters, is_categorical_output)

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

            if max(lowest_error_epoch*1.4,10) < epoch:
                return lowest_error

            if max_epochs is not None and epoch >= max_epochs:
                return lowest_error

            if max_training_time is not None and started_evaluation_at < (int(time.time()) - max_training_time):
                return lowest_error

    def convert_to_device(self,device_str=None):
        if device_str is None:
            device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
        if CONFIG.USE_DEVICE is not None:
            device_str = CONFIG.USE_DEVICE

        device = torch.device(device_str)

        self._mixer.net.to(device)
        self._mixer.net.device = device
        for e in self._mixer.encoders:
            try:
                self._mixer.encoders[e]._model.model.to(device)
                self._mixer.encoders[e]._model.device = device_str
            except:
                pass

    def learn(self, from_data, test_data=None, callback_on_iter = None, eval_every_x_epochs = 20, stop_training_after_seconds=None, stop_model_building_after_seconds=None):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data: (Pandas DataFrame) The data to learn from
        :param test_data: (Pandas DataFrame) The data to test accuracy and learn_error from
        :param callback_on_iter: This is function that can be called on every X evaluation cycle
        :param eval_every_x_epochs: This is every how many epochs we want to calculate the test error and accuracy

        :return: None
        """
        self._stop_training_flag = False

        # This is a helper function that will help us auto-determine roughly what data types are in each column
        # NOTE: That this assumes the data is clean and will only return types for 'CATEGORICAL', 'NUMERIC' and 'TEXT'
        def type_map(col_name):
            col_pd_type =  from_data[col_name].dtype
            col_pd_type = str(col_pd_type)

            if col_pd_type in ['int64', 'float64', 'timedelta']:
                return COLUMN_DATA_TYPES.NUMERIC
            elif col_pd_type in ['bool', 'category']:
                return COLUMN_DATA_TYPES.CATEGORICAL
            else:
                # if the number of uniques is elss than 100 or less than 10% of the total number of rows then keep it as categorical
                unique = from_data[col_name].nunique()
                if  unique < 100 or unique < len(from_data[col_name])/10:
                    return COLUMN_DATA_TYPES.CATEGORICAL
                # else asume its text
                return COLUMN_DATA_TYPES.TEXT

        # generate the configuration and set the order for the input and output columns
        if self._generate_config == True:
            self._input_columns = [col for col in from_data if col not in self._output_columns]
            self.config = {
                'input_features': [{'name': col, 'type': type_map(col)} for col in self._input_columns],
                'output_features': [{'name': col, 'type': type_map(col)} for col in self._output_columns]
            }
            logging.info('Automatically generated a configuration')
            logging.info(self.config)
        else:
            self._output_columns = [col['name'] for col in self.config['input_features']]
            self._input_columns = [col['name'] for col in self.config['output_features']]

        # @TODO Make Cross Entropy Loss work with multiple outputs
        if len(self.config['output_features']) == 1 and self.config['output_features'][0]['type'] in (COLUMN_DATA_TYPES.CATEGORICAL):
            is_categorical_output = True
        else:
            is_categorical_output = False


        if stop_training_after_seconds is None:
            stop_training_after_seconds = round(from_data.shape[0] * from_data.shape[1] / 5)

        if stop_model_building_after_seconds is None:
            stop_model_building_after_seconds = stop_training_after_seconds*3

        from_data_ds = DataSource(from_data, self.config)

        if test_data is not None:
            test_data_ds = DataSource(test_data, self.config)
        else:
            test_data_ds = from_data_ds.extractRandomSubset(0.1)

        from_data_ds.training = True

        mixer_params = {}

        if 'mixer' in self.config:
            mixer_class = self.config['mixer']['class']
            if 'attrs' in  self.config['mixer']:
                mixer_params = self.config['mixer']['attrs']
        else:
            mixer_class = NnMixer

        from_data_ds.prepare_encoders()

        # Initialize data sources
        try:
            mixer_class({}).fit_data_source(from_data_ds)
        except:
            # Not all mixers might require this
            pass

        input_size = len(from_data_ds[0][0])
        training_data_length = len(from_data_ds)

        test_data_ds.transformer = from_data_ds.transformer
        test_data_ds.encoders = from_data_ds.encoders
        test_data_ds.output_weights = from_data_ds.output_weights
        # Initialize data sources

        if input_size != len(test_data_ds[0][0]):
            logging.error("Test and Training dataframe members are of different size !")


        if 'optimizer' in self.config:
            optimizer = self.config['optimizer']()

            while True:
                training_time_per_iteration = stop_model_building_after_seconds/optimizer.total_trials

                # Some heuristics...
                if training_time_per_iteration > input_size:
                    if training_time_per_iteration > min((training_data_length/(4*input_size)), 16*input_size):
                        break

                optimizer.total_trials = optimizer.total_trials - 1
                if optimizer.total_trials < 8:
                    optimizer.total_trials = 8
                    break

            training_time_per_iteration = stop_model_building_after_seconds/optimizer.total_trials

            best_parameters = optimizer.evaluate(lambda dynamic_parameters: Predictor.evaluate_mixer(mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters, is_categorical_output, max_training_time=training_time_per_iteration, max_epochs=None))
            logging.info('Using hyperparameter set: ', best_parameters)
        else:
            # Run a bunch of models through AX and figure out some decent values to put in here
            best_parameters = {}
        mixer = mixer_class(best_parameters, is_categorical_output=is_categorical_output)
        self._mixer = mixer

        for param in mixer_params:
            if hasattr(mixer, param):
                setattr(mixer, param, mixer_params[param])
            else:
                logging.warning('trying to set mixer param {param} but mixerclass {mixerclass} does not have such parameter'.format(param=param, mixerclass=str(type(mixer))))

        eval_next_on_epoch = eval_every_x_epochs
        error_delta_buffer = []  # this is a buffer of the delta of test and train error
        delta_mean = 0
        last_test_error = None
        lowest_error = None
        lowest_error_epoch = None
        last_good_model = None

        started_training_at = int(time.time())
        #iterate over the iter_fit and see what the epoch and mixer error is
        for epoch, training_error in enumerate(mixer.iter_fit(from_data_ds)):
            if self._stop_training_flag == True:
                logging.info('Learn has been stopped')
                break

            logging.info('training iteration {iter_i}, error {error}'.format(iter_i=epoch, error=training_error))

            # see if it needs to be evaluated
            if epoch >= eval_next_on_epoch and test_data_ds:
                tmp_next = eval_next_on_epoch + eval_every_x_epochs
                eval_next_on_epoch = tmp_next

                test_error = mixer.error(test_data_ds)

                # initialize lowest_error_variable if not initialized yet
                if lowest_error is None:
                    lowest_error = test_error
                    lowest_error_epoch = epoch
                    is_lowest_error = True

                else:
                    # define if this is the lowest test error we have had thus far
                    if test_error < lowest_error:
                        lowest_error = test_error
                        lowest_error_epoch = epoch
                        is_lowest_error = True
                    else:
                        is_lowest_error = False

                if last_test_error is None:
                    last_test_error = test_error

                # it its the lowest error, make a FULL copy of the mixer so we can return only the best mixer at the end
                if is_lowest_error:
                    last_good_model = mixer.get_model_copy()

                delta_error = last_test_error - test_error
                last_test_error = test_error

                # keep a stream of training errors delta, so that we can calculate if the mixer is starting to overfit.
                # We assume if the delta of training error starts to increase
                # delta is assumed as the difference between the test and train error
                error_delta_buffer += [delta_error]
                error_delta_buffer = error_delta_buffer[-10:]
                delta_mean = np.mean(error_delta_buffer)
                certainty = mixer.overall_certainty()
                logging.info('certainty {certainty}'.format(certainty=certainty))

                # update mixer and calculate accuracy
                logging.debug('Delta of test error {delta}'.format(delta=delta_mean))

                # if there is a callback function now its the time to call it
                if callback_on_iter is not None:
                    callback_on_iter(epoch, training_error, test_error, delta_mean, self.calculate_accuracy(test_data_ds))



                # Decide if we should stop training
                stop_training = False

                '''
                # Two other potential conditions, not using them for now
                # Stop if the error on the testing data is close to zero
                if test_error < 0.000015:
                    stop_training = True

                # If we've seen no imporvement for a long while, stop
                if lowest_error_epoch + round(max(eval_every_x_epochs*6,epoch*0.5)) < epoch:
                    stop_training = True
                '''

                ## Stop if the model is overfitting
                if delta_mean < 0 and len(error_delta_buffer) > 9:
                    stop_training = True


                # Stop if we're past the time limit alloted for training
                if (int(time.time()) - started_training_at) > stop_training_after_seconds:
                   stop_training = True

                if stop_training:
                    mixer.update_model(last_good_model)
                    self._mixer = mixer
                    self.train_accuracy = self.calculate_accuracy(test_data_ds)
                    self.overall_certainty = certainty
                    break

        # make sure that we update the encoders, we do this, so that the predictor or parent object can pickle the mixers
        self._mixer.encoders = from_data_ds.encoders

        return self


    def predict(self, when_data=None, when=None):
        """
        Predict given when conditions
        :param when_data: a dataframe
        :param when: a dictionary
        :return: a complete dataframe
        """

        if when is not None:
            when_dict = {key:[when[key]] for key in when }
            when_data = pandas.DataFrame(when_dict)

        when_data_ds = DataSource(when_data, self.config)
        when_data_ds.encoders = self._mixer.encoders

        return self._mixer.predict(when_data_ds)

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
        predictions = self._mixer.predict(ds, include_encoded_predictions=True)
        accuracies = {}
        for output_column in self._mixer.output_column_names:
            properties = ds.get_column_config(output_column)
            if properties['type'] == 'categorical':
                accuracies[output_column] = {
                    'function': 'accuracy_score',
                    'value': accuracy_score(list(map(str,ds.get_column_original_data(output_column))), list(map(str,predictions[output_column]["predictions"])))
                }
            else:
                # Note: We use this method instead of using `encoded_predictions` since the values in encoded_predictions are never prefectly 0 or 1, and this leads to rather large unwaranted different in the r2 score, re-encoding the predictions means all "flag" values (sign, isnull, iszero) become either 1 or 0
                encoded_predictions = ds.encoders[output_column].encode(predictions[output_column]["predictions"])
                accuracies[output_column] = {
                    'function': 'r2_score',
                    'value': r2_score(ds.get_encoded_column_data(output_column), encoded_predictions)
                }

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

    def stop_training(self):
        self._stop_training_flag = True
