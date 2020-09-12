
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

            if max_epochs is not None and epoch >= max_epochs:
                return lowest_error

            if max_training_time is not None and started_evaluation_at < (int(time.time()) - max_training_time):
                return lowest_error
