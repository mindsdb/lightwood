from sklearn.metrics import accuracy_score, r2_score, f1_score
from lightwood.constants.lightwood import COLUMN_DATA_TYPES


def _apply_accuracy_function(col_type, reals, preds, weight_map=None, encoder=None):
        if col_type == COLUMN_DATA_TYPES.CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in reals]
            else:
                sample_weight = []
                for val in reals:
                    sample_weight.append(weight_map[val])

            accuracy = {
                'function': 'accuracy_score',
                'value': accuracy_score(reals, preds, sample_weight=sample_weight)
            }
        elif col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
            if weight_map is None:
                sample_weight = [1 for x in reals]
            else:
                sample_weight = []
                for val in reals:
                    sample_weight.append(weight_map[val])

            encoded_reals = encoder.encode(reals)
            encoded_preds = encoder.encode(preds)

            accuracy = {
                'function': 'f1_score',
                'value': f1_score(encoded_reals, encoded_preds, average='weighted', sample_weight=sample_weight)
            }
        else:
            reals_fixed = []
            preds_fixed = []
            for val in reals:
                try:
                    reals_fixed.append(float(val))
                except:
                    reals_fixed.append(0)

            for val in preds:
                try:
                    preds_fixed.append(float(val))
                except:
                    preds_fixed.append(0)

            accuracy = {
                'function': 'r2_score',
                'value': r2_score(reals_fixed, preds_fixed)
            }
        return accuracy


class BaseMixer:
    """
    Base class for all mixers.
    - Overridden __init__ must only accept optional arguments.
    - Subclasses must call BaseMixer.__init__ for proper initialization.
    """
    def __init__(self):
        self.dynamic_parameters = {}

    def fit(self, train_ds, test_ds):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """
        raise NotImplementedError

    def fit_data_source(self, ds):
        """
        :param ds: DataSource
        """
        pass

    def predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """
        raise NotImplementedError

    def evaluate(
        self,
        from_data_ds,
        test_data_ds,
        dynamic_parameters,
        max_training_time=None,
        max_epochs=None
    ):
        """
        :param from_data_ds: DataSource
        :param test_data_ds: DataSource
        :param dynamic_parameters: dict
        :param max_training_time:
        :param max_epochs: int

        :return: float
        """
        self.dynamic_parameters = dynamic_parameters

        started_evaluation_at = int(time.time())
        lowest_error = 10000

        if max_training_time is None and max_epochs is None:
            raise Exception('Please provide either `max_training_time` or `max_epochs`')

        lowest_error_epoch = 0
        for epoch, training_error in enumerate(self.iter_fit(from_data_ds)):
            error = self.error(test_data_ds)

            if lowest_error > error:
                lowest_error = error
                lowest_error_epoch = epoch

            if max(lowest_error_epoch * 1.4, 10) < epoch:
                return lowest_error

            if max_epochs is not None:
                if epoch >= max_epochs:
                    return lowest_error

            if max_training_time is not None:
                if started_evaluation_at < (int(time.time()) - max_training_time):
                    return lowest_error

    def calculate_accuracy(self, ds):
        """
        Calculates the accuracy of the model.

        :param ds: DataSource

        :return: dict of accuracies
        """
        predictions = self.predict(ds, include_extra_data=True)
        accuracies = {}

        for output_column in [feature['name'] for feature in ds.configuration['output_features']]:

            col_type = ds.get_column_config(output_column)['type']

            if col_type == COLUMN_DATA_TYPES.MULTIPLE_CATEGORICAL:
                reals = [tuple(x) for x in ds.get_column_original_data(output_column)]
                preds = [tuple(x) for x in predictions[output_column]['predictions']]
            else:
                reals = [str(x) for x in ds.get_column_original_data(output_column)]
                preds = [str(x) for x in predictions[output_column]['predictions']]

            weight_map = None
            if 'weights' in ds.get_column_config(output_column):
                weight_map = ds.get_column_config(output_column)['weights']

            accuracy = _apply_accuracy_function(
                ds.get_column_config(output_column)['type'],
                reals,
                preds,
                weight_map=weight_map,
                encoder=ds.encoders[output_column]
            )

            if ds.get_column_config(output_column)['type'] == COLUMN_DATA_TYPES.NUMERIC:
                ds.encoders[output_column].decode_log = True
                preds = ds.get_decoded_column_data(
                    output_column,
                    predictions[output_column]['encoded_predictions']
                )

                alternative_accuracy = _apply_accuracy_function(
                    ds.get_column_config(output_column)['type'],
                    reals,
                    preds,
                    weight_map=weight_map
                )

                if alternative_accuracy['value'] > accuracy['value']:
                    accuracy = alternative_accuracy
                else:
                    ds.encoders[output_column].decode_log = False

            accuracies[output_column] = accuracy

        return accuracies