import logging
import traceback
import time

import dill
import copy
import pandas

import numpy as np

from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema

from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
from lightwood.mixers.nn.nn import NnMixer
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, r2_score
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

    def learn(self, from_data, test_data=None, callback_on_iter = None, eval_every_x_epochs = 20, stop_training_after_seconds=3600 * 24 * 5):
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


        mixer = mixer_class()

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
        for epoch, mix_error in enumerate(mixer.iter_fit(from_data_ds)):
            if self._stop_training_flag == True:
                logging.info('Learn has been stopped')
                break

            logging.info('training iteration {iter_i}, error {error}'.format(iter_i=epoch, error=mix_error))

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

                # update mixer and calculate accuracy
                self._mixer = mixer
                accuracy = self.calculate_accuracy(test_data_ds)
                self.train_accuracy = { var: accuracy[var] if accuracy[var] > 0 else 0 for var in accuracy}
                logging.debug('Delta of test error {delta}'.format(delta=delta_mean))

                # if there is a callback function now its the time to call it
                if callback_on_iter is not None:
                    callback_on_iter(epoch, mix_error, test_error, delta_mean)

                # if the model is overfitting that is, that the the test error is becoming greater than the train error
                if (delta_mean < 0 and len(error_delta_buffer) > 5 and test_error < 0.1) or (test_error < 0.005) or (test_error < 0.0005) or (lowest_error_epoch + round(max(eval_every_x_epochs*2+2,epoch*1.2)) < epoch) or ( (int(time.time()) - started_training_at) > stop_training_after_seconds):
                    mixer.update_model(last_good_model)
                    self.train_accuracy = self.calculate_accuracy(test_data_ds)
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
            logging.log.error("Please train the model before calculating accuracy")
            return
        ds = from_data if isinstance(from_data, DataSource) else DataSource(from_data, self.config)
        predictions = self._mixer.predict(ds, include_encoded_predictions=True)
        accuracies = {}
        for output_column in self._mixer.output_column_names:
            properties = ds.get_column_config(output_column)
            if properties['type'] == 'categorical':
                accuracies[output_column] = accuracy_score(ds.get_column_original_data(output_column), predictions[output_column]["predictions"])

            else:
                accuracies[output_column] = r2_score(ds.get_encoded_column_data(output_column), predictions[output_column]["encoded_predictions"])




        return accuracies

    def save(self, path_to):
        """
        save trained model to a file
        :param path_to: full path of file, where we store results
        :return:
        """
        f = open(path_to, 'wb')
        dill.dump(self.__dict__, f)
        f.close()

    def stop_training(self):
        self._stop_training_flag = True
