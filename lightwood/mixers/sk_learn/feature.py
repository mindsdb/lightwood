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
