import unittest

from lightwood.encoders.image.img_2_vec import Img2VecEncoder

class TestDataSource(unittest.TestCase):

    def test_encoding(self):
        # Just some randoms url from imgur's frontpage
        images = ['https://i.imgur.com/PznpPOY.png', 'https://i.imgur.com/B08g7Vk.jpg', 'https://i.imgur.com/WGnlMgh.jpg']

        encoder = Img2VecEncoder()
        encoder.prepare_encoder([])

        ret = encoder.encode(images)

if __name__ == '__main__':
    unittest.main(failfast=True)
