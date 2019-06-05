import torch
from PIL import Image

from lightwood.encoders.image.helpers.img_to_vec import Img2Vec


class Img2VecEncoder:

    def __init__(self, is_target = False):
        self._model = None
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, images):
        """
            Encode list of images

            :images : list of images, each image is a path image(ToDO: url to image also need to be included)
            :return: a torch.floatTensor
        """
        if self._model is None:
            self._model = Img2Vec()

        pics = []
        for image in images:
            img = Image.open(image)
            vec = self._model.get_vec(img)
            pics.append(vec)

        return torch.FloatTensor(pics)


if __name__ == "__main__":
    images = ['test_data/cat.jpg', 'test_data/cat2.jpg', 'test_data/catdog.jpg']

    encoder = Img2VecEncoder()

    ret = encoder.encode(images)
    print(ret)
