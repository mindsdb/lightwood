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
from lightwood.mixers import NnMixer, BaseMixer
from sklearn.metrics import accuracy_score, r2_score, f1_score
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.helpers.device import get_devices


def _apply_accuracy_function(col_type, reals, preds, weight_map=None, encoder=None):
        if col_type == COLUMN_DATA_TYPES.CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in reals]
            else:
                sample_weight = []
                for val in reals:
                    sample_weight.append(weight_map[val])

            accuracy = {
                'function': 'accuracy_score',
                'value': accuracy_score(reals, preds, sample_weight=sample_weight)
            }
        elif col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in reals]
            else:
                sample_weight = []
                for val in reals:
                    sample_weight.append(weight_map[val])

            encoded_reals = encoder.encode(reals)
            encoded_preds = encoder.encode(preds)

            accuracy = {
                'function': 'f1_score',
                'value': f1_score(encoded_reals, encoded_preds, average='weighted', sample_weight=sample_weight)
            }
        else:
            reals_fixed = []
            preds_fixed = []
            for val in reals:
                try:
                    reals_fixed.append(float(val))
                except:
                    reals_fixed.append(0)

            for val in preds:
                try:
                    preds_fixed.append(float(val))
                except:
                    preds_fixed.append(0)

            accuracy = {
                'function': 'r2_score',
                'value': r2_score(reals_fixed, preds_fixed)
            }
        return accuracy


class Predictor:
    def __init__(self, config=None, output=None, load_from_path=None):
        """
        :param config: dict
        :param output: list, the columns you want to predict, ludwig will try to generate a config
        :param load_from_path: str, the path to load the predictor from
        """
        if load_from_path is not None:
            with open(load_from_path, 'rb') as pickle_in:
                self_dict = dill.load(pickle_in)
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

        :param from_data: DataFrame
            The data to learn from
                
        :param test_data: Union[None, DataFrame]
            The data to test accuracy and learn_error from
        """

        # This is a helper function that will help us auto-determine roughly what data types are in each column
        # NOTE: That this assumes the data is clean and will only return types for 'CATEGORICAL', 'NUMERIC' and 'TEXT'
        

        # generate the configuration and set the order for the input and output columns
        if self._generate_config is True:
            self._input_columns = [col for col in from_data if col not in self._output_columns]
            self.config = {
                'input_features': [{'name': col, 'type': self._type_map(from_data, col)} for col in self._input_columns],
                'output_features': [{'name': col, 'type': self._type_map(from_data, col)} for col in self._output_columns]
            }
            self.config = predictor_config_schema.validate(self.config)
            logging.info('Automatically generated a configuration')
            logging.info(self.config)
        else:
            self._output_columns = [col['name'] for col in self.config['output_features']]
            self._input_columns = [col['name'] for col in self.config['input_features']]

        from_data_ds = DataSource(from_data, self.config)

        if test_data is not None:
            test_data_ds = DataSource(test_data, self.config)
        else:
            test_data_ds = from_data_ds.extract_random_subset(0.1)

        from_data_ds.train()

        # Initialize data sources
        if len(from_data_ds) > 100:
            nr_subsets = 3
        else:
            # Don't use k-fold cross validation for very small input sizes
            nr_subsets = 1 

        from_data_ds.prepare_encoders()
        from_data_ds.create_subsets(nr_subsets)

        if 'mixer' in self.config:
            self._mixer = self.config['mixer']
        else:
            self._mixer = NnMixer()

        self._mixer.fit_data_source(from_data_ds)

        test_data_ds.transformer = from_data_ds.transformer
        test_data_ds.encoders = from_data_ds.encoders
        test_data_ds.output_weights = from_data_ds.output_weights
        test_data_ds.create_subsets(nr_subsets)

        self._mixer.fit(train_ds=from_data_ds, test_ds=test_data_ds)

        self.train_accuracy = self.calculate_accuracy(test_data_ds)

        return self

    def predict(self, when_data=None, when=None):
        """
        Predict given when conditions.

        :param when_data: pandas.DataFrame
        :param when: dict

        :return: pandas.DataFrame
        """
        if when is not None:
            when_dict = {key: [when[key]] for key in when}
            when_data = pandas.DataFrame(when_dict)

        when_data_ds = DataSource(when_data, self.config)
        when_data_ds.encoders = self._mixer.encoders

        return self._mixer.predict(when_data_ds)

    def calculate_accuracy(self, from_data):
        """
        Calculates the accuracy of the model.

        :param from_data: pandas.DataFrame

        :return: dict of accuracies
        """

        if self._mixer is None:
            raise Exception('train model before calculating accuracy')

        if isinstance(from_data, DataSource):
            ds = from_data
        else:
            ds = DataSource(from_data, self.config)

        predictions = self._mixer.predict(ds, include_extra_data=True)
        accuracies = {}

        for output_column in self._output_columns:

            col_type = ds.get_column_config(output_column)['type']

            if col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                reals = [tuple(x) for x in ds.get_column_original_data(output_column)]
                preds = [tuple(x) for x in predictions[output_column]['predictions']]
            else:
                reals = [str(x) for x in ds.get_column_original_data(output_column)]
                preds = [str(x) for x in predictions[output_column]['predictions']]

            weight_map = None
            if 'weights' in ds.get_column_config(output_column):
                weight_map = ds.get_column_config(output_column)['weights']

            accuracy = _apply_accuracy_function(
                ds.get_column_config(output_column)['type'],
                reals,
                preds,
                weight_map=weight_map,
                encoder=ds.encoders[output_column]
            )

            if ds.get_column_config(output_column)['type'] == COLUMN_DATA_TYPES.NUMERIC:
                ds.encoders[output_column].decode_log = True
                preds = ds.get_decoded_column_data(
                    output_column,
                    predictions[output_column]['encoded_predictions']
                )

                alternative_accuracy = _apply_accuracy_function(
                    ds.get_column_config(output_column)['type'],
                    reals,
                    preds,
                    weight_map=weight_map
                )

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

            # Dump everything relevant to cpu before saving
            self.convert_to_device("cpu")
            dill.dump(self.__dict__, f)
            self.convert_to_device()
