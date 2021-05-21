from mindsdb_datasources import FileDS
from lightwood import infer_types
from lightwood.api import dtype
import unittest


class TestInferTypes(unittest.TestCase):
    def test_infer_types_on_home_rentlas(self):
        datasource = FileDS('https://raw.githubusercontent.com/mindsdb/mindsdb-examples/master/classics/home_rentals/dataset/train.csv')
        type_information = infer_types(datasource)

        self.assertTrue(type_information.dtypes['number_of_rooms'] == dtype.categorical)
        self.assertTrue(type_information.dtypes['number_of_bathrooms'] == dtype.categorical)
        self.assertTrue(type_information.dtypes['sqft'] == dtype.integer)
        self.assertTrue(type_information.dtypes['location'] == dtype.categorical)
        self.assertTrue(type_information.dtypes['days_on_market'] == dtype.integer)
        self.assertTrue(type_information.dtypes['initial_price'] == dtype.integer)
        self.assertTrue(type_information.dtypes['neighborhood'] == dtype.categorical)
        # This has a .0 after every price, so we shoiuld detect it as float
        self.assertTrue(type_information.dtypes['rental_price'] == dtype.float)

        # There should be no ambiguity in the type detection for this dataset
        for k in type_information.additional_info:
            self.assertTrue(len(type_information.additional_info[k]['dtype_dist']) == 1)

        # None of the columns are identifiers
        for k in type_information.identifiers:
            self.assertTrue(type_information.identifiers[k] is None)
