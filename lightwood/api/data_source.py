import importlib
import numpy as np
from torch.utils.data import Dataset, DataLoader

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

        self.list_cache = {}
        self.encoded_cache = {}
        self.decoded_cache = {}
        self.transformer = None


    def extractRandomSubset(self, percentage):

        msk = np.random.rand(len(self.data_frame)) < (1-percentage)
        test_df = self.data_frame[~msk]
        self.data_frame = self.data_frame[msk]

        return DataSource(test_df, self.configuration)



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


        for feature_set in ['input_features', 'output_features']:
            sample[feature_set] = {}
            for feature in self.configuration[feature_set]:
                col_name = feature['name']
                if col_name not in self.encoded_cache: # if data is not encoded yet, encode values
                    self.get_encoded_column_data(col_name, feature_set)
                sample[feature_set][col_name] = self.encoded_cache[col_name][idx]

        if self.transformer:
            sample = self.transformer.transform(sample)

        return sample

    def get_column_original_data(self, column_name):
        """

        :param column_name:
        :return:
        """

        if column_name in self.list_cache:
            return self.list_cache[column_name]

        if column_name in self.data_frame:
            self.list_cache[column_name] = self.data_frame[column_name].tolist()
            return self.list_cache[column_name]

        else:  # if column not in dataframe
            rows = self.data_frame.shape[0]
            return [None] * rows

    def get_encoded_column_data(self, column_name, feature_set = 'input_features'):
        """

        :param column_name:
        :return:
        """

        if column_name in self.encoded_cache:
            return self.encoded_cache[column_name]

        list_data = self.get_column_original_data(column_name)

        if column_name in self.encoders:
            self.encoded_cache[column_name] = self.encoders[column_name].encode(list_data)

            return self.encoded_cache[column_name]

        config = self.get_column_config(column_name)

        if 'encoder_path' not in config:
            path = 'lightwood.encoders.{type}'.format(type=config['type'])
            module = importlib.import_module(path)
            if hasattr(module, 'default'):
                path += '.' + importlib.import_module(path).default
            else:
                path += '.{type}'.format(type=config['type'])
        else:
            path = config['encoder_path']

        kwargs = config['encoder_args'] if 'encoder_args' in config else {}
        kwargs['is_target'] = True if feature_set == 'output_features' else False
        module = importlib.import_module(path)

        encoder_name = path.split('.')[-1]
        components = encoder_name.split('_')
        encoder_classname = ''.join(x.title() for x in components) + 'Encoder'

        encoder_class = getattr(module, encoder_classname)
        encoder_instance = encoder_class(**kwargs)

        self.encoders[column_name] = encoder_instance

        self.encoded_cache[column_name] = encoder_instance.encode(list_data)

        return self.encoded_cache[column_name]


    def get_decoded_column_data(self, column_name, encoded_data, decoder_instance=None, cache=True):
        """
        :param column_name: column names to be decoded
        :param encoded_data: encoded data of tensor type
        :return decoded_cache : Dict :Decoded data of input column
        """
        if decoder_instance is None:
            if column_name not in self.encoders:
                raise ValueError(
                    'Data must have been encoded before at some point, you should not decode before having encoding at least once')
            decoder_instance = self.encoders[column_name]
        decoded_data = decoder_instance.decode(encoded_data)
        if cache == True:
            self.decoded_cache[column_name] = decoded_data
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
                'type': 'categorical',
                # 'encoder_path': 'lightwood.encoders.categorical.categorical'
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
