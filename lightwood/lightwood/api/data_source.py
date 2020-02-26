import importlib
import inspect
import copy
import random

import numpy as np
from torch.utils.data import Dataset

from lightwood.config.config import CONFIG


class SubSet(Dataset):

    def __init__(self, data_source, indexes):
        self.data_source = data_source
        self.index_mapping = {}
        for i in range(len(indexes)):
            self.index_mapping[i] = indexes[i]

    def __len__(self):
        return int(len(self.index_mapping.keys()))

    def __getitem__(self, idx):
        return self.data_source[self.index_mapping[idx]]

    def get_feature_names(self, where='input_features'):
        return self.data_source.get_feature_names(where)

    def __getattribute__(self, name):
        if name in ['configuration', 'encoders', 'transformer', 'training',
                    'output_weights', 'dropout_dict', 'disable_cache', 'out_types', 'out_indexes']:
            return self.data_source.__getattribute__(name)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name in ['configuration', 'encoders', 'transformer', 'training',
                    'output_weights', 'dropout_dict', 'disable_cache']:
            return dict.__setattr__(self.data_source, name, value)
        else:
            super().__setattr__(name, value)


class DataSource(Dataset):

    def __init__(self, data_frame, configuration):
        """
        Create a lightwood datasource from the data frame
        :param data_frame:
        :param configuration
        """

        self.data_frame = data_frame
        self.configuration = configuration
        self.encoders = {}
        self.transformer = None
        self.training = False  # Flip this flag if you are using the datasource while training
        self.output_weights = None
        self.dropout_dict = {}
        self.enable_dropout = False
        self.disable_cache = not CONFIG.CACHE_ENCODED_DATA
        self.subsets = {}
        self.out_indexes = None
        self.out_types = None

        for col in self.configuration['input_features']:
            if len(self.configuration['input_features']) > 1:
                dropout = 0.1
            else:
                dropout = 0.0

            if 'dropout' in col:
                dropout = col['dropout']

            self.dropout_dict[col['name']] = dropout

        self._clear_cache()

    def create_subsets(self, nr_subsets):
        subsets_indexes = {}
        np.random.seed(len(self.data_frame))

        subset_nr = 1
        for i in range(len(self.data_frame)):
            if subset_nr not in subsets_indexes:
                subsets_indexes[subset_nr] = []
            subsets_indexes[subset_nr].append(i)

            subset_nr += 1
            if subset_nr > nr_subsets:
                subset_nr = 1

        for subset_nr in subsets_indexes:
            self.subsets[subset_nr] = SubSet(self, subsets_indexes[subset_nr])

    def _clear_cache(self):
        self.encoded_cache = {}
        self.transformed_cache = None

    def extractRandomSubset(self, percentage):
        np.random.seed(int(round(percentage * 100000)))
        msk = np.random.rand(len(self.data_frame)) < (1 - percentage)
        test_df = self.data_frame[~msk]
        self.data_frame = self.data_frame[msk]
        # clear caches
        self._clear_cache()

        ds = DataSource(test_df, self.configuration)
        ds.encoders = self.encoders
        ds.transformer = self.transformer
        return ds

    def __len__(self):
        """
        return the length of the datasource (as in number of rows)
        :return: number of rows
        """
        return int(self.data_frame.shape[0])

    def __getitem__(self, idx):
        """

        :param idx:
        :return:
        """
        sample = {}

        dropout_features = None

        if self.training == True and random.randint(0,3) == 1 and self.enable_dropout and CONFIG.ENABLE_DROPOUT:
            dropout_features = [feature['name'] for feature in self.configuration['input_features'] if random.random() > (1 - self.dropout_dict[feature['name']])]

            # Make sure we never drop all the features, since this would make the row meaningless
            if len(dropout_features) > len(self.configuration['input_features']):
                dropout_features = dropout_features[:-1]
            #logging.debug(f'\n-------------\nDroping out features: {dropout_features}\n-------------\n')

        if self.transformed_cache is None and not self.disable_cache:
            self.transformed_cache = [None] * self.__len__()

        if not self.disable_cache and not (dropout_features is not None and len(dropout_features) > 0):
            cached_sample = self.transformed_cache[idx]
            if cached_sample is not None:
                return cached_sample

        for feature_set in ['input_features', 'output_features']:
            sample[feature_set] = {}
            for feature in self.configuration[feature_set]:
                col_name = feature['name']
                col_config = self.get_column_config(col_name)
                if col_name not in self.encoded_cache:  # if data is not encoded yet, encode values
                    if not ((dropout_features is not None and col_name in dropout_features) or self.disable_cache):
                        self.get_encoded_column_data(col_name)

                # if we are dropping this feature, get the encoded value of None
                if dropout_features is not None and col_name in dropout_features:
                    custom_data = {col_name: [None]}
                    # if the dropout feature depends on another column, also pass a None array as the dependant column
                    if 'depends_on_column' in col_config:
                        custom_data[custom_data['depends_on_column']] = [None]
                    sample[feature_set][col_name] = self.get_encoded_column_data(col_name, custom_data=custom_data)[0]
                elif self.disable_cache:
                    if col_name in self.data_frame:
                        custom_data = {col_name: [self.data_frame[col_name].iloc[idx]]}
                    else:
                        custom_data = {col_name: [None]}

                    sample[feature_set][col_name] = self.get_encoded_column_data(col_name, custom_data=custom_data)[0]
                else:
                    sample[feature_set][col_name] = self.encoded_cache[col_name][idx]

        # Create weights if not already create
        if self.output_weights is None:
            for col_config in self.configuration['output_features']:
                if 'weights' in col_config:

                    weights = col_config['weights']
                    new_weights = None

                    for val in weights:
                        encoded_val = self.get_encoded_column_data(
                            col_config['name'], custom_data={col_config['name']: [val]})
                        # @Note: This assumes one-hot encoding for the encoded_value
                        value_index = np.argmax(encoded_val[0])

                        if new_weights is None:
                            new_weights = [np.mean(list(weights.values()))] * len(encoded_val[0])

                        new_weights[value_index] = weights[val]

                    if self.output_weights is None:
                        self.output_weights = new_weights
                    else:
                        self.output_weights.extend(new_weights)
                else:
                    self.output_weights = False

        if self.transformer:
            sample = self.transformer.transform(sample)
            if self.out_indexes is None:
                self.out_indexes = self.transformer.out_indexes

        if not self.disable_cache:
            self.transformed_cache[idx] = sample
            return self.transformed_cache[idx]
        else:
            return sample

    def get_column_original_data(self, column_name):
        """

        :param column_name:
        :return:
        """
        if column_name not in self.data_frame:
            nr_rows = self.data_frame.shape[0]
            return [None] * nr_rows

        return self.data_frame[column_name].tolist()


    def prepare_encoders(self):
        """
            Get the encoder for all the output and input column and preapre them
            with all available data for that column.

            * Note: This method should only be called on the "main" training dataset, all
            the other datasets should get their encoders and transformers
            from the training dataset.
        """
        input_encoder_training_data = {'targets': []}
        self.out_types = []

        for feature_set in ['output_features', 'input_features']:
            for feature in self.configuration[feature_set]:
                column_name = feature['name']
                config = self.get_column_config(column_name)

                args = [self.get_column_original_data(column_name)]

                # If the column depends upon another, it's encoding *might* be influenced by that
                if 'depends_on_column' in config:
                    args += [self.get_column_original_data(config['depends_on_column'])]

                # If the encoder is not specified by the user lookup the default encoder for the column's data type
                if 'encoder_class' not in config:
                    path = 'lightwood.encoders.{type}'.format(type=config['type'])
                    module = importlib.import_module(path)
                    if hasattr(module, 'default'):
                        encoder_class = importlib.import_module(path).default
                    else:
                        raise ValueError('No default encoder for {type}'.format(type=config['type']))
                else:
                    encoder_class = config['encoder_class']

                # Instantiate the encoder and pass any arguments given via the configuration
                is_target = True if feature_set == 'output_features' else False
                encoder_instance = encoder_class(is_target=is_target)

                if is_target:
                    self.out_types.append(config['type'])

                encoder_attrs = config['encoder_attrs'] if 'encoder_attrs' in config else {}
                for attr in encoder_attrs:
                    if hasattr(encoder_instance, attr):
                        setattr(encoder_instance, attr, encoder_attrs[attr])

                # Prime the encoder using the data (for example, to get the one-hot mapping in a categorical encoder)
                if feature_set == 'input_features':
                    training_data = input_encoder_training_data
                else:
                    training_data = None

                if 'training_data' in inspect.getargspec(encoder_instance.prepare_encoder).args:
                    encoder_instance.prepare_encoder(args[0], training_data=training_data)
                else:
                    encoder_instance.prepare_encoder(args[0])

                self.encoders[column_name] = encoder_instance

                if feature_set == 'output_features':
                    input_encoder_training_data['targets'].append({
                        'encoded_output': copy.deepcopy(self.encoders[column_name].encode(args[0])),
                        'unencoded_output': copy.deepcopy(args[0]),
                        'output_encoder': copy.deepcopy(encoder_instance),
                        'output_type': copy.deepcopy(config['type'])
                    })

        return True

    def get_encoded_column_data(self, column_name, custom_data=None):
        """

        :param column_name:
        :return:
        """

        if column_name in self.encoded_cache and custom_data is None:
            return self.encoded_cache[column_name]

        # The first argument of encoder is the data, if no custom data is specified,
        # use all the datasource's data for this column
        if custom_data is not None:
            args = [custom_data[column_name]]
        else:
            args = [self.get_column_original_data(column_name)]

        config = self.get_column_config(column_name)

        # See if the feature has dependencies in other columns
        if 'depends_on_column' in config:
            if custom_data is not None:
                arg2 = custom_data[config['depends_on_column']]
            else:
                arg2 = self.get_column_original_data(config['depends_on_column'])
            args += [arg2]

        if column_name in self.encoders:
            encoded_vals = self.encoders[column_name].encode(*args)
            # Cache the encoded data so we don't have to run the encoding,
            # Don't cache custom_data
            # (custom_data is usually used when running without cache or dropping out a feature for a certain pass)
            if column_name not in self.encoded_cache and custom_data is None:
                self.encoded_cache[column_name] = encoded_vals
            return encoded_vals
        else:
            raise Exception(
                'Looks like you are trying to encode data before preating the encoders via calling `prepare_encoders`')

    def get_decoded_column_data(self, column_name, encoded_data, decoder_instance=None):
        """
        :param column_name: column names to be decoded
        :param encoded_data: encoded data of tensor type
        :return decoded_data : Dict :Decoded data of input column
        """
        if decoder_instance is None:
            if column_name not in self.encoders:
                raise ValueError("""
                    Data must have been encoded before at some point.
                    You should not decode before having encoding at least once
                    """)
            decoder_instance = self.encoders[column_name]
        decoded_data = decoder_instance.decode(encoded_data)

        return decoded_data

    def get_feature_names(self, where='input_features'):

        return [feature['name'] for feature in self.configuration[where]]

    def get_column_config(self, column_name):
        """
        Get the config info for the feature given a configuration as defined in data_schemas definition.py
        :param column_name:
        :return:
        """
        for feature_set in ['input_features', 'output_features']:
            for feature in self.configuration[feature_set]:
                if feature['name'] == column_name:
                    return feature


if __name__ == "__main__":
    #TODO; sometimes this fail depending on the data generated
    import random
    import pandas

    config = {
        'name': 'test',
        'input_features': [
            {
                'name': 'x',
                'type': 'numeric',

            },
            {
                'name': 'y',
                'type': 'numeric',

            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',

            }
        ]
    }

    data = {'x': [i for i in range(10)], 'y': [random.randint(i, i + 20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i < 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    print(data_frame)

    ds = DataSource(data_frame, config)
    ds.prepare_encoders()
    print(ds.get_encoded_column_data('z'))
