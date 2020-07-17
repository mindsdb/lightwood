import logging
import os

import torch

from lightwood.encoders.image.helpers.nn import NnEncoderHelper
from lightwood.encoders.encoder_base import BaseEncoder


class NnAutoEncoder(BaseEncoder):

    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._model = None

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._model = NnEncoderHelper(images)
        self._prepared = True

    def encode(self, images):
        """
          Encode all the images from the list of paths(to images)

        :param images: List of images paths
        :return: a torch.floatTensor
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        if not self._model:
            logging.error("No model to encode, please train the model")

        return self._model.encode(images)

    def decode(self, encoded_values_tensor, save_to_path="decoded/"):
        """
         Decoded the encoded list of image tensors and write the decoded images to give path

        :param encoded_values_tensor: List of encoded images tensors
        :param save_to_path: Path to store decoded images
        :return: a list of image paths
        """
        if not self._model:
            logging.error("No model to decode, please train the model")

        if not os.path.exists(save_to_path):
            os.makedirs(save_to_path)
        return self._model.decode(encoded_values_tensor, save_to_path)

    def train(self, images):
        """
        :param images: List of images paths
        """
        self._model = NnEncoderHelper(images)


# only run the test if this file is called from debugger
if __name__ == "__main__":
    #TODO: add images
    images = ['test_data/cat.jpg', 'test_data/cat2.jpg', 'test_data/catdog.jpg']
    encoder = NnAutoEncoder(images)

    encoder.prepare_encoder([])
    images = ['test_data/cat.jpg', 'test_data/cat2.jpg']
    encoded_data = encoder.encode(images)
    print(encoded_data)

    # decoded images will be stored under decoded folder
    decoded_data = encoder.decode(encoded_data, "decoded/images")
    print(decoded_data)
