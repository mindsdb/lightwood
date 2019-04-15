
class ColumnAbstract:

    def __init__(self, type):
        """
        This initializes the column, and has a pointer to the parent dataset,
        which includes the dataframe as well as the predictor

        :param column_name: the column that we are analyzing
        :param parent_dataset: the parent dataset
        """

        self._type = None # This column is
        self._encoded_status = None # None, Encoding, Done
        pass

    def encode(self):
        """
        This is meant to run asynchronously, and once its done encoding it should change teh self._encoded status to Done
        :return: None
        """
        pass


    def decode(self):
        """

        :return:
        """
        pass