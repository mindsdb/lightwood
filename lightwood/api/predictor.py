import logging
import traceback

import dill

from lightwood.api.data_source import DataSource
from lightwood.data_schemas.definition import definition_schema

from lightwood.mixers.sk_learn.sk_learn import SkLearnMixer
from lightwood.mixers.nn.nn import NnMixer
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, r2_score


class Predictor:

    def __init__(self, definition=None, load_from_path=None):
        """
        Start a predictor pass the

        :param definition: a predictor definition object (can be a dictionary or a PredictorDefinition object)
        :param load_from_path: The path to load the predictor from
        :type definition: dictionary
        """

        if load_from_path is not None:
            pickle_in = open(load_from_path, "rb")
            self_dict = dill.load(pickle_in)
            pickle_in.close()
            self.__dict__ = self_dict
            return

        try:
            definition_schema.validate(definition)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        self.definition = definition
        self._encoders = None
        self._mixer = None
        self._mixers = {}
        self.train_accuracy = None

    def learn(self, from_data, test_data=None, validation_data=None):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data:
        :param test_data:
        :param validation_data:
        :return:
        """

        from_data_ds = DataSource(from_data, self.definition)
        if test_data:
            test_data_ds = DataSource(test_data, self.definition)
        else:
            test_data_ds = from_data_ds.extractRandomSubset(0.1)

        default_mixer_params = {}

        if 'default_mixer' in self.definition:
            default_mixer_class = self.definition['default_mixer']['class']
            if 'params' in  self.definition['default_mixer']:
                default_mixer_params = self.definition['default_mixer']['params']
        else:
            default_mixer_class = NnMixer



        default_mixer_args = {}
        default_mixer_args['input_column_names'] = [f['name'] for f in self.definition['input_features']]
        default_mixer_args['output_column_names'] = [f['name'] for f in self.definition['output_features']]

        mixer = default_mixer_class(**default_mixer_args)

        for param in default_mixer_params:
            if hasattr(mixer, param):
                setattr(mixer, param, default_mixer_params[param])
            else:
                logging.warning('trying to set mixer param {param} but mixerclass {mixerclass} does not have such parameter'.format(param=param, mixerclass=str(type(mixer))))

        for i, mix_error in enumerate(mixer.iter_fit(from_data_ds)):
            logging.info('training iteration {iter_i}, error {error}'.format(iter_i=i, error=mix_error))

        self._mixer = mixer
        self._mixer.encoders = from_data_ds.encoders

        self.train_accuracy =  self.accuracy(test_data_ds)




    def predict(self, when_data):
        """
        Predict given when conditions
        :param when_data: a dataframe
        :return: a complete dataframe
        """

        when_data_ds = DataSource(when_data, self.definition)
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
        ds = from_data if isinstance(from_data, DataSource) else DataSource(from_data, self.definition)
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
        ],

        'default_mixer': {
            'class': NnMixer,
            'params': {
                'epochs': 1000,
                'criterion': nn.MSELoss(),
                'optimizer_args' : {'lr': 0.01}
            }
        }

        # 'default_mixer': {
        #     'class': SkLearnMixer
        # }
    }

    datapoints = 10000

    data = {'x': [random.randint(-10,10) for i in range(datapoints)], 'y': [random.randint(-10,10) for i in range(datapoints)]}
    nums = [data['x'][i] * data['y'][i] for i in range(datapoints)]
    data['z'] = [data['x'][i] * data['y'][i]  for i in range(datapoints)]
    #data['z2'] = [data['x'][i] * data['y'][i] for i in range(100)]
    data_frame = pandas.DataFrame(data)

    print(data_frame)

    ####################
    predictor = Predictor(definition=config)
    predictor.learn(from_data=data_frame)
    print(predictor.train_accuracy)
    print(predictor.accuracy(from_data=data_frame))
    print(predictor.predict(when_data=pandas.DataFrame({'x': [1], 'y': [0]})))
    predictor.save('tmp\ok.pkl')

    predictor2 = Predictor(load_from_path='tmp\ok.pkl')
    print(predictor2.predict(when_data=pandas.DataFrame({'x': [0, 0, 1, -1, 1], 'y': [0, 1, -1, -1, 1]})))
    print(predictor2.predict(when_data=pandas.DataFrame({'x': [0, 3, 1, -5, 1], 'y': [0, 1, -5, -4, 7]})))
