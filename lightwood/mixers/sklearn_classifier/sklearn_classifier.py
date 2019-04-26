
class SklearnClassifier:

    def __init__(self, input_column_names, output_column_names ):
        """
        :param input_column_names: is a list [col_name1, col_name2]
        :param output_column_names: is a list [col_name1, col_name2]
        """


        self.input_column_names = input_column_names
        self.output_column_names = output_column_names

        self.model = None

        pass


    def fit(self, data_source):
        '''
        :param data: is a DataSource object
        :return:
        '''

        pass


    def predict(self, when_data_source):
        '''

        :param when_data: is a DataSource object
        :return:
        '''

        pass






if __name__ == "__main__":
    import random
    import pandas
    from lightwood.api.data_source import DataSource


    ###############
    # GENERATE DATA
    ###############

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
                #'encoder_path': 'lightwood.encoders.numeric.numeric'
            }
        ],

        'output_features': [
            {
                'name': 'z',
                'type': 'categorical',
                #'encoder_path': 'lightwood.encoders.categorical.categorical'
            }
        ]
    }

    data = {'x':[i for i in range(10)], 'y':[random.randint(i,i+20) for i in range(10)]}
    nums = [data['x'][i] * data['y'][i] for i in range(10)]

    data['z'] = ['low' if i< 50 else 'high' for i in nums]

    data_frame = pandas.DataFrame(data)

    print(data_frame)

    ds = DataSource(data_frame, config)
    ####################



    mixer = SklearnClassifier(input_column_names = ['x', 'y'], output_column_names=['y'])

    mixer.fit(data)

    

