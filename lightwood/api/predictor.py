import traceback
import time

import pandas
import numpy as np
import torch


from lightwood.api.data_source import DataSource
from lightwood.data_schemas.predictor_config import predictor_config_schema
from lightwood.config.config import CONFIG
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices
from lightwood.logger import log


class Predictor:
    def __init__(self, config=None, output=None, load_from_path=None):
        """
        :param config: dict
        :param output: list, the columns you want to predict, ludwig will try to generate a config
        :param load_from_path: str, the path to load the predictor from
        """
        if load_from_path is not None:
            with open(load_from_path, 'rb') as pickle_in:
                self_dict = torch.load(pickle_in)
            self.__dict__ = self_dict
            self.convert_to_device()
            return

        if output is None and config is None:
            raise ValueError('You must provide either `output` or `config`')

        if config is not None and output is None:
            try:
                self.config = predictor_config_schema.validate(config)
            except Exception:
                error = traceback.format_exc(1)
                raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        # this is if we need to automatically generate a configuration variable
        self._generate_config = True if output is not None or self.config is None else False

        self._output_columns = output
        self._input_columns = None
        self.train_accuracy = None

        self._mixer = None

    def convert_to_device(self, device_str=None):
        if hasattr(self._mixer, 'to') and callable(self._mixer.to):
            if device_str is not None:
                device = torch.device(device_str)
                available_devices = 1
                if device_str == 'cuda':
                    available_devices = torch.cuda.device_count()
            else:
                device, available_devices = get_devices()

            self._mixer.to(device, available_devices)

    def _type_map(self, from_data, col_name):
        """
        This is a helper function that will help us auto-determine roughly what data types are in each column
        NOTE: That this assumes the data is clean and will only return types for 'CATEGORICAL', 'NUMERIC' and 'TEXT'
        """

        col_pd_type = from_data[col_name].dtype
        col_pd_type = str(col_pd_type)

        if col_pd_type in ['int64', 'float64', 'timedelta']:
            return COLUMN_DATA_TYPES.NUMERIC
        elif col_pd_type in ['bool', 'category']:
            return COLUMN_DATA_TYPES.CATEGORICAL
        else:
            # if the number of uniques is less than 100 or less
            # than 10% of the total number of rows then keep it as categorical
            unique = from_data[col_name].nunique()
            if unique < 100 or unique < len(from_data[col_name]) / 10:
                return COLUMN_DATA_TYPES.CATEGORICAL
            else:
                return COLUMN_DATA_TYPES.TEXT

    def learn(self, from_data, test_data=None):
        """
        Train and save a model (you can use this to retrain model from data).

        :param from_data: DataFrame or DataSource
            The data to learn from

        :param test_data: DataFrame or DataSource
            The data to test accuracy and learn_error from
        """
        device, _available_devices = get_devices()
        log.info(f'Computing device used: {device}')
        # generate the configuration and set the order for the input and output columns
        if self._generate_config is True:
            self._input_columns = [col for col in from_data if col not in self._output_columns]
            self.config = {
                'input_features': [{'name': col, 'type': self._type_map(from_data, col)} for col in self._input_columns],
                'output_features': [{'name': col, 'type': self._type_map(from_data, col)} for col in self._output_columns]
            }
            self.config = predictor_config_schema.validate(self.config)
            log.info('Automatically generated a configuration')
            log.info(self.config)
        else:
            self._output_columns = [col['name'] for col in self.config['output_features']]
            self._input_columns = [col['name'] for col in self.config['input_features']]

        if isinstance(from_data, pandas.DataFrame):
            train_ds = DataSource(from_data, self.config)
        elif isinstance(from_data, DataSource):
            train_ds = from_data
        else:
            raise TypeError(':from_data: must be either DataFrame or DataSource')

        nr_subsets = 3 if len(train_ds) > 100 else 1

        if test_data is None:
            test_ds = train_ds.subset(0.1)
        elif isinstance(test_data, pandas.DataFrame):
            test_ds = train_ds.make_child(test_data)
        elif isinstance(test_data, DataSource):
            test_ds = test_data
        else:
            raise TypeError(':test_data: must be either DataFrame or DataSource')

        train_ds.create_subsets(nr_subsets)
        test_ds.create_subsets(nr_subsets)

        train_ds.train()
        test_ds.train()

        mixer_class = self.config['mixer']['class']
        mixer_kwargs = self.config['mixer']['kwargs']
        self._mixer = mixer_class(**mixer_kwargs)
        self._mixer.fit(train_ds=train_ds, test_ds=test_ds)
        self.train_accuracy = self._mixer.calculate_accuracy(test_ds)

        return self

    def predict(self, when_data=None, when=None):
        """
        Predict given when conditions.

        :param when_data: pandas.DataFrame
        :param when: dict

        :return: pandas.DataFrame
        """
        device, _available_devices = get_devices()
        log.info(f'Computing device used: {device}')
        if when is not None:
            when_dict = {key: [when[key]] for key in when}
            when_data = pandas.DataFrame(when_dict)

        when_data_ds = DataSource(when_data, self.config, prepare_encoders=False)

        when_data_ds.eval()

        kwargs = {'include_extra_data': self.config.get('include_extra_data', False)}

        return self._mixer.predict(when_data_ds, **kwargs)

    def calculate_accuracy(self, from_data):
        """
        calculates the accuracy of the model
        :param from_data:a dataframe
        :return accuracies: dictionaries of accuracies
        """

        if self._mixer is None:
            log.error("Please train the model before calculating accuracy")
            return

        ds = from_data if isinstance(from_data, DataSource) else DataSource(from_data, self.config, prepare_encoders=False)
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
        Save trained model to a file.

        :param path_to: str, full path of file, where we store results
        """
        with open(path_to, 'wb') as f:
            # Null out certain object we don't want to store
            if hasattr(self._mixer, '_nonpersistent'):
                self._mixer._nonpersistent = {}
            self.config['mixer']['kwargs']['callback_on_iter'] = None


            # Dump everything relevant to cpu before saving
            self.convert_to_device("cpu")
            torch.save(self.__dict__, f)
            self.convert_to_device()
