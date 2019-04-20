import os

import torch
from PIL import Image

from lightwood.column_data_types.image.helpers.img_to_vec import Img2Vec


class Img2VecEncoder:

    def __init__(self):
        self._model = None

    def encode(self, path):
        """
            Encode all images present under a directory

            :param path: directory where images present
            :return: a torch.floatTensor
        """
        if self._model is None:
            self._model = Img2Vec()

        pics = []
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(path, filename))
            vec = self._model.get_vec(img)
            pics.append(vec)

        return torch.FloatTensor(pics)


if __name__ == "__main__":
    input_path = 'test_data/'
    encoder = Img2VecEncoder()
    ret = encoder.encode(input_path)
    print(ret)
