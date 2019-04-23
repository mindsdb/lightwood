import traceback

from lightwood.data_schemas.definition import definition_schema
from lightwood.constants.lightwood import COLUMN_DATA_TYPES, HISTOGRAM_TYPES


class Predictor:



    def __init__(self, definition):
        """
        Start a predictor pass the

        :param definition: a predictor definition object (can be a dictionary or a PredictorDefinition object)
        :type definition: dictionary
        """
        try:
            definition_schema.validate(definition)
        except:
            error = traceback.format_exc(1)
            raise ValueError('[BAD DEFINITION] argument has errors: {err}'.format(err=error))

        self.definition = definition
        self._model = None
        self._encoders = None


    def learn(self, from_data, test_data, validation_data):
        """
        Train and save a model (you can use this to retrain model from data)

        :param from_data:
        :param test_data:
        :param validation_data:
        :return:
        """

        model, encoders = self._load()

        validation_dataset = Dataset(df=validation_data, parent_predictor_object=self)
        test_dataset = Dataset(df=test_data, parent_predictor_object=self)

        train_dataset = Dataset(df=from_data, parent_predictor_object=self)
        train_dataset.load_encoders(encoders) # if encoders exist start from there
        train_dataset.encode(validation_dataset=validation_dataset)

        if model is None: # build model if no model exists
            model = Model(definition=self.definition)

        model.train(train_dataset, validation_dataset = test_dataset)

        self._save(model = model, encoders = train_dataset.get_encoders())


    def predict(self, when_data):
        """
        Predict given when conditions
        :param when: a dataframe
        :return: a complete dataframe
        """

        model, encoders = self._load()

        when_dataset = Dataset(df=when_data, parent_predictor_object=self)
        when_dataset.load_encoders(encoders)
        when_dataset.encode(train_encoders = False)

        complete_dataset = model.forward(when_dataset)

        return complete_dataset.get_df()


    def _save(self, model, encoders):
        """

        :param model:
        :param encoders:
        :return:
        """

        pass

    def _load(self, use_cache = True):
        """

        :param use_cache: (default True) if the model already in memory use it
        :return: model, encoders
        """
        model = self._model
        encoders = self._encoders

        return model, encoders




# only run the test if this file is called from debugger
if __name__ == "__main__":
    Predictor(definition = {'name':'Will', 'input_features': [{'name':'trio', 'type': COLUMN_DATA_TYPES.TEXT}]} )

