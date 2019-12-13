import importlib
import inspect
import random
import copy

import numpy as np
from torch.utils.data import Dataset

from lightwood.config.config import CONFIG


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
        self.training = False # Flip this flag if you are using the datasource while training
        self.output_weights = None
        self.dropout_dict = {}
        self.disable_cache = not CONFIG.CACHE_ENCODED_DATA

        for col in self.configuration['input_features']:
            if len(self.configuration['input_features']) > 1:
                dropout = 0.0
            else:
                dropout = 0.0

            if 'dropout' in col:
                dropout = col['dropout']

            self.dropout_dict[col['name']] = dropout

        self._clear_cache()


    def _clear_cache(self):
        self.list_cache = {}
        self.encoded_cache = {}
        self.transformed_cache = None


    def extractRandomSubset(self, percentage):
        np.random.seed(int(round(percentage*100000)))
        msk = np.random.rand(len(self.data_frame)) < (1-percentage)
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

        if self.training == True and random.randint(0,2) == 1:
            dropout_features = [feature['name'] for feature in self.configuration['input_features'] if random.random() > (1 - self.dropout_dict[feature['name']])]

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
                if col_name not in self.encoded_cache: # if data is not encoded yet, encode values
                    if not ((dropout_features is not None and  col_name in dropout_features) or self.disable_cache):
                        self.get_encoded_column_data(col_name)


                # if we are dropping this feature, get the encoded value of None
                if dropout_features is not None and col_name in dropout_features:
                    custom_data = {col_name:[None]}
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
                        encoded_val = self.get_encoded_column_data(col_config['name'],custom_data={col_config['name']:[val]})
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

        if self.disable_cache:
            return self.data_frame[column_name].tolist()

        elif column_name in self.list_cache:
            return self.list_cache[column_name]

        else:
            self.list_cache[column_name] = self.data_frame[column_name].tolist()
            return self.list_cache[column_name]

    def prepare_encoders(self):
        '''
            Get the encoder for all the output and input column and preapre them
            with all available data for that column.

            * Note: This method should only be called on the "main" training dataset, all
            the other datasets should get their encoders and transformers
            from the training dataset.
        '''
        input_encoder_training_data = {'targets': []}

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
                        'encoded_output': copy.deepcopy(self.encoders[column_name].encode(args[0]))
                        ,'unencoded_output': copy.deepcopy(args[0])
                        ,'output_encoder': copy.deepcopy(encoder_instance)
                        ,'output_type': copy.deepcopy(config['type'])
                    })

        return True


    def get_encoded_column_data(self, column_name, custom_data = None):
        """

        :param column_name:
        :return:
        """

        if column_name in self.encoded_cache and custom_data is None:
            return self.encoded_cache[column_name]

        # The first argument of encoder is the data, if no custom data is specified, use all the datasource's data for this column
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
            # Don't cache custom_data (custom_data is usually used when running without cache or dropping out a feature for a certain pass)
            if column_name not in self.encoded_cache and custom_data is None:
                self.encoded_cache[column_name] = encoded_vals
            return encoded_vals
        else:
            raise Exception('It looks like you are trying to encode data before preating the encoders via calling `prepare_encoders`')


    def get_decoded_column_data(self, column_name, encoded_data, decoder_instance=None):
        """
        :param column_name: column names to be decoded
        :param encoded_data: encoded data of tensor type
        :return decoded_data : Dict :Decoded data of input column
        """
        if decoder_instance is None:
            if column_name not in self.encoders:
                raise ValueError('Data must have been encoded before at some point, you should not decode before having encoding at least once')
            decoder_instance = self.encoders[column_name]
        decoded_data = decoder_instance.decode(encoded_data)

        return decoded_data

    def get_feature_names(self, where = 'input_features'):

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

    print(ds.get_encoded_column_data('z'))
