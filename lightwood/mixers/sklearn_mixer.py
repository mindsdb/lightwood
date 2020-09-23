from lightwood.mixers import BaseMixer


classification = [
    SGDClassifier
    SVC
    NaiveBayes
]




class SklearnMixer(BaseMixer):
    """
    Selects model based on the following cheat-sheet:
    https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
    """

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
