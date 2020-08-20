import lightwood
import unittest
import math

if lightwood.encoders.export_cesium:
    from lightwood.encoders import CesiumTsEncoder


    class TestCesiumTsEncoder(unittest.TestCase):
        def test_encoder(self):

            data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

            ret = CesiumTsEncoder(features=[
                "amplitude",
                "percent_beyond_1_std",
                "maximum",
                "max_slope",
                "median",
                "median_absolute_deviation",
                "percent_close_to_median",
                "minimum",
                "skew",
                "std",
                "weighted_average"
            ]).encode(data)

            print(ret)
