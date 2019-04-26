import importlib

class DataSource:

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



    def getColumnData(self, column_name):

        if column_name in self.list_cache:
            return self.list_cache[column_name]

        if column_name in self.data_frame:
            self.list_cache[column_name] =  self.data_frame[column_name].tolist()
            return self.list_cache[column_name]

        else: # if column not in dataframe
            rows = self.data_frame.shape[0]
            return [None]*rows




    def getEncodedColumnData(self, column_name):

        if column_name in self.encoded_cache:

            return self.encoded_cache[column_name]

        list_data = self.getColumnData(column_name)

        config = self._getColumnConfig(column_name)

        path = config['encoder_path']
        kwargs = config['encoder_args'] if 'encoder_args' in config else {}

        module = importlib.import_module(path)

        encoder_name = path.split('.')[-1]
        components = encoder_name.split('_')
        encoder_classname = ''.join(x.title() for x in components)+'Encoder'

        encoder_class = getattr(module, encoder_classname)
        encoder_instance = encoder_class(**kwargs)

        self.encoders[column_name] = encoder_instance

        self.encoded_cache[column_name] = encoder_instance.encode(list_data)

        return self.encoded_cache[column_name]


    def _getColumnConfig(self, column_name):
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
                'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',
                'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x':[i for i in range(10)], 'y':[random.randint(i,i+20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i< 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    print(data_frame)

    ds = DataSource(data_frame, config)

    print(ds.getEncodedColumnData('z'))

