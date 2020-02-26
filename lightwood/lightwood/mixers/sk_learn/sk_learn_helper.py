import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler


class SkLearnMixerHelper:

    def _input_encoded_columns(self, target_column, when_data_source):
        """
        :param when_data_source: is a DataSource object
        :return: numpy.nd array input encoded values
        """
        input_encoded = None
        for column in self.feature_columns[target_column]:
            if input_encoded is None:
                input_encoded = self._encoded_data([column], when_data_source)
            else:
                input_encoded = np.append(input_encoded, self._encoded_data([column], when_data_source), axis=1)
        return StandardScaler().fit_transform(input_encoded)

    def _output_encoded_columns(self, column, data_source):
        """
        :param data_source: is a DataSource object
        :return: numpy.nd array output encoded values
        """
        output_encoded_column = self._encoded_data([column], data_source)
        self.output_encoders = data_source.encoders
        return output_encoded_column

    def _extract_features(self, data_source, model_class, output_encoded_column):
        """
        :param data_source: is a DataSource object
        :param model_class: type of model to be fitted
        :param output_encoded_column: target variable encoded values
        :return: numpy.nd array: important input encoded columns
        """
        input_encoded_columns = None
        feature_columns = []
        for column in self.input_column_names:
            input_encoded_column = self._encoded_data([column], data_source)
            input_encoded_column = StandardScaler().fit_transform(input_encoded_column)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_class.fit(StandardScaler().fit_transform(input_encoded_column), output_encoded_column)
            score = model.score(input_encoded_column, output_encoded_column)
            self.feature_models[column] = model
            self.feature_importance[column] = score

            if score > self.score_threshold:
                feature_columns.append(column)
                if input_encoded_columns is None:
                    input_encoded_columns = input_encoded_column
                else:
                    input_encoded_columns = np.append(input_encoded_columns, input_encoded_column, axis=1)
        return StandardScaler().fit_transform(input_encoded_columns), feature_columns

    @staticmethod
    def _encoded_data(features, data_source):
        """
        :param features: list of column names
        :param data_source: input data
        :return encoded_data: numpy.nd array encoded values
        """
        for cnt, column in enumerate(features):
            if cnt == 0:
                encoded_data = data_source.get_encoded_column_data(column).numpy()
            else:
                encoded_data = np.append(encoded_data, data_source.get_encoded_column_data(column).numpy(), axis=1)
        return encoded_data

    def _decoded_data(self, features, data_source, encoded_data):
        """
        :param features: list : columns to be decoded
        :param data_source: is a DataSource object
        :param encoded_data: encoded data
        :return:  decoded data
        """
        for column in features:
            encoders = self.output_encoders.get(column, None)
            if encoders is None:
                decoded_data = data_source.get_decoded_column_data(column, encoded_data)
            else:
                decoded_data = encoders.decode(encoded_data)
        return decoded_data

    def _determine_model_class(self, column, data_source):
        """
        :param column: name of the column
        :param data_source: is a DataSource object
        :return: model: Model to be considered for fitting data
        """
        data_type = data_source.get_column_config(column)['type']
        from lightwood.mixers.sk_learn.feature import FeatureFactory

        if data_type is not None:
            feature = FeatureFactory.create_feature(data_source.get_column_config(column))
            model = feature.get_model_class(self.classifier_class, self.regression_class)
            return model
