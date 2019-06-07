import logging
import traceback

import dill
import copy

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

        self._generate_config = True if output is not None else False
        self._output_columns = output
        self._input_columns = None
        self._encoders = None
        self._mixer = None
        self._mixers = {}
        self.train_accuracy = None

    def learn(self, from_data, test_data=None, validation_data=None, callback_on_iter = None):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data:
        :param test_data:
        :param validation_data:
        :return:
        """

        def type_map(col_pd_type):

            if col_pd_type in ['int64', 'float64', 'timedelta']:
                return COLUMN_DATA_TYPES.NUMERIC
            elif col_pd_type in ['bool', 'category']:
                return COLUMN_DATA_TYPES.CATEGORICAL
            else:
                return COLUMN_DATA_TYPES.TEXT

        if self._generate_config == True:
            self._input_columns = [col for col in from_data if col not in self._output_columns]
            self.config = {
                'input_features': [{'name': col, 'type': type_map(from_data[col].dtype)} for col in self._input_columns],
                'output_features': [{'name': col, 'type': type_map(from_data[col].dtype)} for col in self._output_columns]
            }
            logging.info('Automatically generated a configuration')
            logging.info(self.config)
        else:
            self._output_columns = [col['name'] for col in self.config['input_features']]
            self._input_columns = [col['name'] for col in self.config['output_features']]


        from_data_ds = DataSource(from_data, self.config)
        if test_data:
            test_data_ds = DataSource(test_data, self.config)
        else:
            test_data_ds = from_data_ds.extractRandomSubset(0.1)

        default_mixer_params = {}

        if 'default_mixer' in self.config:
            default_mixer_class = self.config['default_mixer']['class']
            if 'attrs' in  self.config['default_mixer']:
                default_mixer_params = self.config['default_mixer']['attrs']
        else:
            default_mixer_class = NnMixer



        default_mixer_args = {}
        default_mixer_args['input_column_names'] = [f['name'] for f in self.config['input_features']]
        default_mixer_args['output_column_names'] = [f['name'] for f in self.config['output_features']]

        mixer = default_mixer_class(**default_mixer_args)

        for param in default_mixer_params:
            if hasattr(mixer, param):
                setattr(mixer, param, default_mixer_params[param])
            else:
                logging.warning('trying to set mixer param {param} but mixerclass {mixerclass} does not have such parameter'.format(param=param, mixerclass=str(type(mixer))))



        epoch_eval_jump = 1000
        eval_next_on_epoch = epoch_eval_jump

        error_delta_buffer = []  # this is a buffer of the delta of test and train error
        delta_mean = 0
        last_test_error = None
        lowest_error = None
        last_good_model = None

        for epoch, mix_error in enumerate(mixer.iter_fit(from_data_ds)):
            logging.info('training iteration {iter_i}, error {error}'.format(iter_i=epoch, error=mix_error))

            if epoch >= eval_next_on_epoch and test_data_ds:

                tmp_next = eval_next_on_epoch + epoch_eval_jump
                eval_next_on_epoch = tmp_next

                test_error = mixer.error(test_data_ds)
                if lowest_error is None:
                    lowest_error = test_error
                    is_lowest_error = True

                else:
                    if test_error < lowest_error:
                        lowest_error = test_error
                        is_lowest_error = True
                    else:
                        is_lowest_error = False

                if last_test_error is None:
                    last_test_error = test_error

                if is_lowest_error:
                    last_good_model = mixer.get_model_copy()

                delta_error = last_test_error - test_error
                last_test_error = test_error

                error_delta_buffer += [delta_error]
                error_delta_buffer = error_delta_buffer[-10:]
                delta_mean = np.mean(error_delta_buffer)
                logging.debug('Delta of test error {delta}'.format(delta=delta_mean))

                if callback_on_iter is not None:
                    callback_on_iter(epoch, mix_error, test_error, delta_mean, self)

                if delta_mean < 0:
                    mixer.update_model(last_good_model)
                    break





        self._mixer = mixer
        self._mixer.encoders = from_data_ds.encoders

        self.train_accuracy =  self.accuracy(test_data_ds)




    def predict(self, when_data=None, when=None):
        """
        Predict given when conditions
        :param when_data: a dataframe
        :param when: a dictionary
        :return: a complete dataframe
        """

        if when is not None:

            when_data = pandas.DataFrame(when)

        when_data_ds = DataSource(when_data, self.config)
        when_data_ds.encoders = self._mixer.encoders

        return self._mixer.predict(when_data_ds)

    def accuracy(self, from_data):
        """
        calculates the accuracy of the model
        :param from_data:a dataframe
        :return accuracies: dictionaries of accuracies
        """
        if self._mixer is None:
            logging.log.error("Please train the model before calculating accuracy")
            return
        ds = from_data if isinstance(from_data, DataSource) else DataSource(from_data, self.config)
        predictions = self._mixer.predict(ds)
        accuracies = {}
        for output_column in self._mixer.output_column_names:
            properties = ds.get_column_config(output_column)
            if properties['type'] == 'categorical':
                accuracies[output_column] = accuracy_score(ds.get_column_original_data(output_column), predictions[output_column]["Actual Predictions"])

            else:
                accuracies[output_column] = r2_score(ds.get_encoded_column_data(output_column), predictions[output_column]["Encoded Predictions"])




        return {'accuracies': accuracies}

    def save(self, path_to):
        """
        save trained model to a file
        :param path_to: full path of file, where we store results
        :return:
        """
        f = open(path_to, 'wb')
        dill.dump(self.__dict__, f)
        f.close()


# only run the test if this file is called from debugger
if __name__ == "__main__":
    # GENERATE DATA
    ###############
    import pandas
    import random
    import torch.nn as nn

    config = {

        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',
                #'encoder_path': 'lightwood.encoders.numeric.numeric'
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
        ],

        'default_mixer': {
            'class': NnMixer,
            'attrs': {
                'epochs': 2000,
                'criterion': nn.MSELoss(),
                'optimizer_args' : {'lr': 0.01}
            }
        }

        # 'default_mixer': {
        #     'class': SkLearnMixer
        # }
    }

    datapoints = 100

    data = {'x': [random.randint(-10,10) for i in range(datapoints)], 'y': [random.randint(-10,10) for i in range(datapoints)]}
    nums = [data['x'][i] * data['y'][i] for i in range(datapoints)]
    data['z'] = [data['x'][i] * data['y'][i]  for i in range(datapoints)]
    #data['z2'] = [data['x'][i] * data['y'][i] for i in range(100)]
    data_frame = pandas.DataFrame(data)

    print(data_frame)

    ####################
    predictor = Predictor(output=['z'])
    def feedback(iter, error, test_error, test_error_gradient, predictor):
        print('iteration: {iter}, error: {error}, test_error: {test_error}, test_error_gradient: {test_error_gradient}'.format(iter=iter, error=error, test_error=test_error, test_error_gradient=test_error_gradient))


    predictor.learn(from_data=data_frame, callback_on_iter=feedback )
    print(predictor.train_accuracy)
    print(predictor.accuracy(from_data=data_frame))
    print(predictor.predict(when_data=pandas.DataFrame({'x': [1], 'y': [0]})))
    predictor.save('tmp\ok.pkl')

    predictor2 = Predictor(load_from_path='tmp\ok.pkl')
    print(predictor2.predict(when_data=pandas.DataFrame({'x': [0, 0, 1, -1, 1], 'y': [0, 1, -1, -1, 1]})))
    print(predictor2.predict(when_data=pandas.DataFrame({'x': [0, 3, 1, -5, 1], 'y': [0, 1, -5, -4, 7]})))
    