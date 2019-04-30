import logging

import numpy as np
import torch
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class SkLearnClassifier:

    def __init__(self, input_column_names, output_column_names):
        """
        :param input_column_names: is a list [col_name1, col_name2]
        :param output_column_names: is a list [col_name1, col_name2]
        """
        self.input_column_names = input_column_names
        self.output_column_names = output_column_names
        self.feature_columns = []
        self.output_encoders = {}
        self.model = None

    def fit(self, data_source):
        '''
        :param data_source: is a DataSource object
        :return model: fitted model
        '''
        logging.info('Model training started')
        input_encoded_columns = None
        output_encoded_column = self._encoded_data(self.output_column_names, data_source)
        self.output_encoders = data_source.encoders
        for column in self.input_column_names:
            input_encoded_column = self._encoded_data([column], data_source)
            input_encoded_column = StandardScaler().fit_transform(input_encoded_column)
            model = MultiOutputClassifier(KNeighborsClassifier(3), n_jobs=-1).fit(input_encoded_column,
                                                                                  output_encoded_column)
            data = (input_encoded_column, output_encoded_column)
            score = self._cal_score(data, model)
            if score > 0.5:
                self.feature_columns.append(column)
                if input_encoded_columns is None:
                    input_encoded_columns = input_encoded_column
                else:
                    np.append(input_encoded_columns, input_encoded_column, axis=1)
        input_encoded_columns = StandardScaler().fit_transform(input_encoded_columns)

        self.model = MultiOutputClassifier(KNeighborsClassifier(3), n_jobs=-1).fit(input_encoded_columns,
                                                                                   output_encoded_column)
        data = (input_encoded_columns, output_encoded_column)
        model_score = self._cal_score(data, model)
        logging.info('Model training completed with score:{}'.format(model_score))
        return self.model

    def predict(self, when_data_source):
        '''
        :param when_data_source: is a DataSource object
        :return predictions: numpy.ndarray predicted encoded values
        '''
        logging.info('Model predictions starting')
        input_encoded = None
        when_data_source.encoders = self.output_encoders
        for column in self.feature_columns:
            if input_encoded is None:
                input_encoded = self._encoded_data([column], when_data_source)
            else:
                np.append(input_encoded, self._encoded_data([column], when_data_source))
        input_encoded = StandardScaler().fit_transform(input_encoded)
        encoded_predictions = self.model.predict(input_encoded)
        decoded_predictions = self._decoded_data(self.output_column_names, when_data_source,
                                                 torch.from_numpy(encoded_predictions))
        logging.info('Model predictions and decoding completed')
        return {'Encoded Predictions': encoded_predictions,
                'Actual Predictions ': decoded_predictions}

    def _encoded_data(self, features, data_source):
        """
        :param features: list of column names
        :param data_source: input data
        :return encoded_data: numpy.ndarray encoded values
        """
        for cnt, column in enumerate(features):
            if cnt == 0:
                encoded_data = data_source.getEncodedColumnData(column).numpy()
            else:
                np.append(encoded_data, data_source.getEncodedColumnData(column).numpy(), axis=1)
        return encoded_data

    def _decoded_data(self, features, data_source, data):
        """
        :param features: list : columns to be decoded
        :param data_source: is a DataSource object
        :param data: encoded data
        :return:  decoded data
        """
        for column in features:
            decoded_data = data_source.decoded_column_data(column, data)
        return decoded_data

    def _cal_score(self, data, model):
        """
        :param data: input data
        :param model: trained model
        :return score: score calculated using input data
        """
        return model.score(data[0], data[1])


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

    # print(data_frame)

    ds = DataSource(data_frame, config)
    predict_input_ds = DataSource(data_frame[['x', 'y']], config)
    ####################

    mixer = SkLearnClassifier(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predictions = mixer.predict(predict_input_ds)
    print(predictions)
