import torch
from PIL import Image

from lightwood.encoders.image.helpers.img_to_vec import Img2Vec


class Img2VecEncoder:

    def __init__(self):
        self._model = None
        # I think we should make this an enum, something like: fast, balanced, accurate
        self.speed = 'balanced'
        self._pytorch_wrapper = torch.FloatTensor

    def encode(self, images):
        """
            Encode list of images

            :images : list of images, each image is a path image(ToDO: url to image also need to be included)
            :return: a torch.floatTensor
        """
        if self._model is None:
            if self.speed == 'fast':
                self._model = Img2Vec(model='alexnet')
            elif self.speed == 'balanced':
                self._model = Img2Vec(model='resnet-18')
            elif self.speed == 'accurate':
                self._model = Img2Vec(model='resnext-50')
            else:
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
