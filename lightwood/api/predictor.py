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
                predictor_config_schema.validate(config)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        self.config = config

        # this is if we need to automatically generate a configuration variable
        self._generate_config = True if output is not None or self.config is None else False

        self._output_columns = output
        self._input_columns = None

        self._mixer = None
        self._helper_mixers = None

        self.train_accuracy = None
        self.overall_certainty = None

    @staticmethod
    def evaluate_mixer(mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters,
                       max_training_time=None, max_epochs=None):
        started_evaluation_at = int(time.time())
        lowest_error = 10000
        mixer = mixer_class(dynamic_parameters)

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

    def train_helper_mixers(self, train_ds, test_ds):
        from lightwood.mixers.boost.boost import BoostMixer

        boost_mixer = BoostMixer()
        boost_mixer.train(train_ds)

        # @TODO: IF we add more mixers in the future, add the best on for each column to this map !
        best_mixer_map = {}
        predictions = boost_mixer.predict(test_ds)

        for output_column in self._output_columns:
            model = boost_mixer.targets[output_column]['model']
            if model is None:
                continue

            real = list(map(str,test_ds.get_column_original_data(output_column)))
            predicted =  predictions[output_column]['values']

            weight_map = None
            if 'weights' in test_ds.get_column_config(output_column):
                weight_map = train_ds.get_column_config(output_column)['weights']

            accuracy = self.apply_accuracy_function(train_ds.get_column_config(output_column)['type'], real, predicted, weight_map)
            best_mixer_map[output_column] = {
                'model': boost_mixer
                ,'accuracy': accuracy['value']
            }
        return best_mixer_map


    def learn(self, from_data, test_data=None, callback_on_iter = None, eval_every_x_epochs = 20, stop_training_after_seconds=None, stop_model_building_after_seconds=None):
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
        nr_subsets = 3
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

            best_parameters = optimizer.evaluate(lambda dynamic_parameters: Predictor.evaluate_mixer(
                mixer_class, mixer_params, from_data_ds, test_data_ds, dynamic_parameters,
                max_training_time=training_time_per_iteration, max_epochs=None))

            logging.info('Using hyperparameter set: ', best_parameters)
        else:
            best_parameters = {}

        if CONFIG.HELPER_MIXERS and self.has_boosting_mixer and (CONFIG.FORCE_HELPER_MIXERS or len(from_data_ds) < 12 * pow(10,3)):
            try:
                self._helper_mixers = self.train_helper_mixers(from_data_ds, test_data_ds)
            except Exception as e:
                logging.warning(f'Failed to train helper mixers with error: {e}')

        mixer = mixer_class(best_parameters)
        self._mixer = mixer

        for param in mixer_params:
            if hasattr(mixer, param):
                setattr(mixer, param, mixer_params[param])
            else:
                logging.warning(
                    'trying to set mixer param {param} but mixerclass {mixerclass} does not have such parameter'.format
                    (param=param, mixerclass=str(type(mixer)))
                )

        started = time.time()
        log_reasure = time.time()
        epoch = 0
        eval_next_on_epoch = eval_every_x_epochs
        first_run = True
        stop_training = False

        for subset_iteration in [1, 2]:
            if stop_training:
                break
            subset_id_arr =  [*from_data_ds.subsets.keys()] # [1]
            for subset_id in subset_id_arr:
                started_subset = time.time()
                if stop_training:
                    break

                #subset_train_ds = from_data_ds #.subsets[subset_id]
                #subset_test_ds = test_data_ds #.subsets[subset_id]

                subset_train_ds = from_data_ds.subsets[subset_id]
                subset_test_ds = test_data_ds.subsets[subset_id]

                lowest_error = None
                last_test_error = None
                last_subset_test_error = None
                test_error_delta_buff = []
                subset_test_error_delta_buff = []
                best_model = None
                best_selfaware_model = None

                #iterate over the iter_fit and see what the epoch and mixer error is
                for epoch, training_error in enumerate(mixer.iter_fit(subset_train_ds, initialize=first_run, subset_id=subset_id)):
                    first_run = False

                    # Log this every now and then so that the user knows it's running
                    if (int(time.time()) - log_reasure) > 30:
                        log_reasure = time.time()
                        logging.info(f'Lightwood training, iteration {epoch}, training error {training_error}')


                    # Prime the model on each subset for a bit
                    if subset_iteration == 1:
                        break

                    # Once the training error is getting smaller, enable dropout to teach the network to predict without certain features
                    if subset_iteration == 2 and training_error < 0.4 and not from_data_ds.enable_dropout:
                        eval_every_x_epochs = max(1, int(eval_every_x_epochs/2) )
                        logging.info('Enabled dropout !')
                        from_data_ds.enable_dropout = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    # If the selfaware network isn't able to train, go back to the original network
                    if subset_iteration == 2 and (np.isnan(training_error) or np.isinf(training_error) or training_error > pow(10,5)):
                        mixer.start_selfaware_training = False
                        mixer.stop_selfaware_training = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    # Once we are past the priming/warmup period, start training the selfaware network
                    if subset_iteration == 2 and not mixer.is_selfaware and CONFIG.SELFAWARE and not mixer.stop_selfaware_training and training_error < 0.35:
                        logging.info('Started selfaware training !')
                        mixer.start_selfaware_training = True
                        lowest_error = None
                        last_test_error = None
                        last_subset_test_error = None
                        test_error_delta_buff = []
                        subset_test_error_delta_buff = []
                        continue

                    if epoch >= eval_next_on_epoch:
                        eval_next_on_epoch += eval_every_x_epochs

                        test_error = mixer.error(test_data_ds)
                        subset_test_error = mixer.error(subset_test_ds, subset_id=subset_id)
                        logging.info(f'Subtest test error: {subset_test_error} on subset {subset_id}')

                        if lowest_error is None or test_error < lowest_error:
                            lowest_error = test_error
                            if mixer.is_selfaware:
                                best_selfaware_model = mixer.get_model_copy()
                            else:
                                best_model = mixer.get_model_copy()

                        if last_subset_test_error is None:
                            pass
                        else:
                            subset_test_error_delta_buff.append(last_subset_test_error - subset_test_error)

                        last_subset_test_error = subset_test_error

                        if last_test_error is None:
                            pass
                        else:
                            test_error_delta_buff.append(last_test_error - test_error)

                        last_test_error = test_error

                        delta_mean = np.mean(test_error_delta_buff[-5:])
                        subset_delta_mean = np.mean(subset_test_error_delta_buff[-5:])

                        if callback_on_iter is not None:
                            callback_on_iter(epoch, training_error, test_error, delta_mean,
                                             self.calculate_accuracy(test_data_ds))

                        ## Stop if the model is overfitting
                        #if delta_mean <= 0 and len(test_error_delta_buff) > 4:
                        #    stop_training = True

                        # Stop if we're past the time limit allocated for training
                        if (time.time() - started) > stop_training_after_seconds:
                            stop_training = True

                        # If the trauining subset is overfitting on it's associated testing subset
                        if (subset_delta_mean <= 0 and len(subset_test_error_delta_buff) > 4) or (time.time() - started_subset) > stop_training_after_seconds/len(from_data_ds.subsets.keys()):
                            logging.info('Finished fitting on {subset_id} of {no_subsets} subset'.format(subset_id=subset_id, no_subsets=len(from_data_ds.subsets.keys())))

                            if mixer.is_selfaware:
                                if best_selfaware_model is not None:
                                    mixer.update_model(best_selfaware_model)
                            else:
                                mixer.update_model(best_model)


                            if subset_id == subset_id_arr[-1]:
                                stop_training = True
                            elif not stop_training:
                                break

                        if stop_training:
                            if mixer.is_selfaware:
                                mixer.update_model(best_selfaware_model)
                            else:
                                mixer.update_model(best_model)
                            self._mixer = mixer
                            self.train_accuracy = self.calculate_accuracy(test_data_ds)
                            self.overall_certainty = mixer.overall_certainty()
                            logging.info('Finished training model !')
                            break

        self._mixer.build_confidence_normalization_data(test_data_ds)
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
            when_dict = {key: [when[key]] for key in when}
            when_data = pandas.DataFrame(when_dict)

        when_data_ds = DataSource(when_data, self.config)
        when_data_ds.encoders = self._mixer.encoders

        main_mixer_predictions = self._mixer.predict(when_data_ds)

        if CONFIG.HELPER_MIXERS and self.has_boosting_mixer:
            for output_column in main_mixer_predictions:
                if self._helper_mixers is not None and output_column in self._helper_mixers:
                    if self._helper_mixers[output_column]['accuracy'] > 1.00 * self.train_accuracy[output_column]['value']:
                        helper_mixer_predictions = self._helper_mixers[output_column]['model'].predict(when_data_ds, output_column)
                        main_mixer_predictions[output_column] = {'predictions': list(helper_mixer_predictions[output_column]['values'])}
                        if 'confidences' in helper_mixer_predictions[output_column] and helper_mixer_predictions[output_column]['confidences'] is not None:
                            main_mixer_predictions[output_column]['confidences'] = list(helper_mixer_predictions[output_column]['confidences'])

        return main_mixer_predictions

    @staticmethod
    def apply_accuracy_function(col_type, real, predicted, weight_map=None):
        if col_type == 'categorical':
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
        predictions = self._mixer.predict(ds, include_encoded_predictions=True)
        accuracies = {}

        for output_column in self._output_columns:

            real = list(map(str,ds.get_column_original_data(output_column)))
            predicted =  list(map(str,predictions[output_column]['predictions']))

            weight_map = None
            if 'weights' in ds.get_column_config(output_column):
                weight_map = ds.get_column_config(output_column)['weights']

            accuracy = self.apply_accuracy_function(ds.get_column_config(output_column)['type'], real, predicted,weight_map=weight_map)

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
