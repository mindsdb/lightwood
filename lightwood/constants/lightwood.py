# Guideline (make sure this file is pep-8 compliant)
# create a class and the create a constant alias for the class


class Const:

    def get_attributes(self):
        a = {val:self.__getattribute__(val) for val in dir(self) if val[0]!='_' and val[0].isupper()}
        return a


class ColumnDataTypes(Const):

    NUMERIC = 'numeric'
    CATEGORICAL = 'categorical'
    DATETIME = 'datetime'
    IMAGE = 'image'
    TEXT = 'text'
    TIME_SERIES = 'time_series'

COLUMN_DATA_TYPES = ColumnDataTypes()


class HistogramTypes(Const):

    NORMAL = 'normal'
    EXPONENTIAL = 'exponential'

HISTOGRAM_TYPES = HistogramTypes()


class EncoderAim(Const):

    SPEED = 'speed'
    BALANCE = 'balance'
    ACCURACY = 'accuracy'

ENCODER_AIM = EncoderAim()
