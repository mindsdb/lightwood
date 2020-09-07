
class BaseMixer:
    def __init__(self, config=None):
        self.config = config
        self.dynamic_parameters = {}
    
    def set_dynamic_parameters(self, dynamic_parameters):
        self.dynamic_parameters = dynamic_parameters

    def fit(self, train_ds, test_ds, callback, stop_training_after_seconds, eval_every_x_epochs):
        raise NotImplementedError

    def iter_fit(self, *args, **kwargs):
        raise NotImplementedError

    def fit_data_source(self, *args, **kwargs):
        pass

    def predict(self):
        raise NotImplementedError

    def evaluate(self, from_data_ds, test_data_ds, dynamic_parameters,
                 max_training_time=None, max_epochs=None):
        self.set_dynamic_parameters(dynamic_parameters)

        started_evaluation_at = int(time.time())
        lowest_error = 10000

        if max_training_time is None and max_epochs is None:
            err = "Please provide either `max_training_time` or `max_epochs` when calling `evaluate`"
            logging.error(err)
            raise Exception(err)

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
