from mindsdb_datasources import FileDS
from lightwood import infer_types
from lightwood.api import dtype
import unittest


class TestInferTypes(unittest.TestCase):
    def test_infer_types_on_home_rentlas(self):
        datasource = FileDS('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/home_rentals/dataset/train.csv')
        type_information = infer_types(datasource)
        print(type_information)
