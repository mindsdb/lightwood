from lightwood.mixers import BaseMixer

class SklearnMixer(BaseMixer):
    def __init__(self):
        super().__init__()

    def fit(self, train_ds, test_ds):
        """
        :param train_ds: DataSource
        :param test_ds: DataSource
        """
        self.fit_datasource(train_ds)
        raise NotImplementedError
    
    def predict(self, when_data_source, include_extra_data=False):
        """
        :param when_data_source: DataSource
        :param include_extra_data: bool
        """
        raise NotImplementedError
