
class BaseMixer:
    def __init__(self, ds):
        """
        params
            ds: DataSource
        """
        pass

    def fit(self, train_ds, test_ds, callback, stop_training_after_seconds, eval_every_x_epochs):
        pass

    def predict(self):
        pass
