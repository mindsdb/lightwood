import numpy as np
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
        self.model = None

    def fit(self, data_source):
        '''
        :param data_source: is a DataSource object
        :return:
        '''
        input_encoded = None
        output_encoded = self._encoded_data(self.output_column_names, data_source)
        for column in self.input_column_names:
            input_single_encoded = self._encoded_data(column, data_source)
            input_single_encoded = StandardScaler().fit_transform(input_single_encoded)
            model = MultiOutputClassifier(KNeighborsClassifier(3), n_jobs=-1).fit(input_single_encoded,
                                                                                  output_encoded)
            data = (input_single_encoded, output_encoded)
            score = self._cal_score(data, model)
            if score > 0.5:
                self.feature_columns.append(column)
                if input_encoded is None:
                    input_encoded = input_single_encoded
                else:
                    np.append(input_encoded, input_single_encoded, axis=1)
        input_encoded = StandardScaler().fit_transform(input_encoded)
        self.model = MultiOutputClassifier(KNeighborsClassifier(3), n_jobs=-1).fit(input_encoded,
                                                                                   output_encoded)
        data = (input_encoded, output_encoded)
        model_score = self._cal_score(data, model)
        print(model_score)
        return self.model

    def predict(self, when_data_source):
        '''
        :param when_data: is a DataSource object
        :return:
        '''
        input_encoded = None
        for column in self.feature_columns:
            if input_encoded is None:
                input_encoded = self._encoded_data(column, when_data_source)
            else:
                np.append(input_encoded, self._encoded_data(column, when_data_source))
        input_encoded = StandardScaler().fit_transform(input_encoded)
        predictions = self.model.predict(input_encoded)
        return predictions

    def _encoded_data(self, feature, data_source):
        for cnt, column in enumerate(feature):
            if cnt == 0:
                encoded_data = data_source.getEncodedColumnData(column).numpy()
            else:
                np.append(encoded_data, data_source.getEncodedColumnData(feature).numpy(), axis=1)
        return encoded_data

    def _cal_score(self, data, model):
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
    ####################

    mixer = SkLearnClassifier(input_column_names=['x', 'y'], output_column_names=['z'])

    data_encoded = mixer.fit(ds)
    predict_encoded = mixer.predict(ds)
    print(predict_encoded)
