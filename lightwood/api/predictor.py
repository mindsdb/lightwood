import traceback

from lightwood.api.data_source import DataSource
from lightwood.constants.lightwood import COLUMN_DATA_TYPES
from lightwood.data_schemas.definition import definition_schema
from lightwood.mixers.sklearn_mixer.sklearn_mixer import SKLearnMixer


class Predictor:

    def __init__(self, definition, load_from_path=None):
        """
        Start a predictor pass the

        :param definition: a predictor definition object (can be a dictionary or a PredictorDefinition object)
        :param load_from_path: The path to load the predictor from
        :type definition: dictionary
        """
        try:
            definition_schema.validate(definition)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        self.definition = definition
        self._encoders = None
        self._mixers = None

    def learn(self, from_data, test_data=None, validation_data=None):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data:
        :param test_data:
        :param validation_data:
        :return:
        """

        from_data_ds = DataSource(from_data, self.definition)
        if test_data:
            test_data_ds = DataSource(test_data, self.definition)
        else:
            test_data_ds = None

        if len(self.definition['output_features']) == 1:
            model = SKLearnMixer(
                input_column_names=[f['name'] for f in self.definition['input_features']],
                output_column_names=[f['name'] for f in self.definition['output_features']])

            model.fit(from_data_ds)

        pass

    def predict(self, when_data):
        """
        Predict given when conditions
        :param when: a dataframe
        :return: a complete dataframe
        """

        pass

    def save(self, path_to):
        """

        :param path:
        :return:
        """

        pass


# only run the test if this file is called from debugger
if __name__ == "__main__":
    Predictor(definition={'name': 'Will', 'input_features': [{'name': 'trio', 'type': COLUMN_DATA_TYPES.TEXT}]})
