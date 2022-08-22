import unittest

from lightwood.api.dtype import dtype
from lightwood.data.infer_types import type_check_date


class TestTypeInference(unittest.TestCase):
    def test_0_type_check_dates(self):
        self.assertEqual(type_check_date('31/12/2010'), dtype.date)
