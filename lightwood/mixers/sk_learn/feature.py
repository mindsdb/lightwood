from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


class CategoricalFeature:

    def __init__(self, properties):
        self.type = properties['type']
        self.name = properties['name']

    def get_model_class(self, classifier_class, regression_class):
        """

        :param classifier_class: 
        :param regression_class: 
        :return:  model which will be used to fit the data
        """
        return classifier_class(KNeighborsClassifier(3), n_jobs=-1)

    def calculate_accuracy(self, ds, predictions):
        """

        :param ds: actual data set
        :param predictions: Predicted data set
        :return:  accuracy
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(ds.get_column_original_data(self.name),
                              predictions[self.name]["Actual Predictions"])


class NumericFeature:

    def __init__(self, properties):
        self.type = properties['type']
        self.name = properties['name']

    def get_model_class(self, classifier_class, regression_class):
        """

        :param classifier_class: 
        :param regression_class: 
        :return: 
        """
        return regression_class(svm.SVR())

    def calculate_accuracy(self, ds, predictions):
        """

        :param ds: actual data set
        :param predictions: Predicted data set
        :return:  accuracy
        """
        from sklearn.metrics import explained_variance_score

        return explained_variance_score(ds.get_encoded_column_data(self.name),
                                        predictions[self.name]["Encoded Predictions"])


class FeatureFactory:

    @staticmethod
    def create_feature(properties):
        """

        :param properties:
        :return:
        """
        if properties['type'] == 'categorical':
            return CategoricalFeature(properties)
        elif properties['type'] == 'numeric':
            return NumericFeature(properties)
