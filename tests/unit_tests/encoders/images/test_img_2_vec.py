import unittest

from lightwood.encoders.image.img_2_vec import Img2VecEncoder

class TestDataSource(unittest.TestCase):

    def test_encoding(self):
        pass
        # # Just some randoms url from imgur's frontpage
        # # @TODO: See how this behaves for SVGs
        # images = ['https://i.imgur.com/PznpPOY.png', 'https://i.imgur.com/B08g7Vk.jpg', 'https://i.imgur.com/WGnlMgh.jpg']

        # encoder = Img2VecEncoder()
        # encoder.prepare(images)

        # ret = encoder.encode(images)

        # self.assertTrue(len(ret.shape) == 2)
        # self.assertTrue(ret.shape[0] == len(images))
