# Guideline (make sure this file is pep-8 compliant)
# create a class and the create a constant alias for the class


class Const:

    def get_attributes(self):
        a = {val:self.__getattribute__(val) for val in dir(self) if val[0]!='_' and val[0].isupper()}
        return a

class ColumnDataTypes(Const):

    INT = 'int'
    FLOAT = 'float'

    DATE_STRING = 'date_string'
    TIMESTAMP = 'timestamp'
    TIMESTAMP_DELTA = 'timestamp_delta'

    BINARY_IMAGE = 'binary_image'
    BINDARY_AUDIO = 'binary_audio'

    PATH_TO_IMAGE = 'path_to_image'
    PATH_TO_AUDIO = 'path_to_audio'

    URL_TO_IMAGE = 'url_to_image'
    URL_TO_AUDIO = 'url_to_audio'

    SINGLE_CATEGORY = 'single_category'
    MULTIPLE_CATEGORIES = 'multiple_categories'

    TEXT = 'text'
    LIST_OF_NUMBERS = 'list_of_numbers'

COLUMN_DATA_TYPES = ColumnDataTypes() # this is to make sure we are pep8 compliant



class HistogramTypes(Const):

    NORMAL = 'normal'
    EXPONENTIAL = 'exponential'

HISTOGRAM_TYPES = HistogramTypes()


