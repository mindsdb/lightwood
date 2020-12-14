from io import BytesIO
import logging

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, TiffImagePlugin, PngImagePlugin
import requests

from lightwood.encoders.image.helpers.img_to_vec import Img2Vec
from lightwood.constants.lightwood import ENCODER_AIM
from lightwood.encoders.encoder_base import BaseEncoder


class Img2VecEncoder(BaseEncoder):

    def __init__(self, is_target=False, aim=ENCODER_AIM.SPEED):
        super().__init__(is_target)
        self.model = None
        # I think we should make this an enum, something like: speed, balance, accuracy
        self.aim = aim
        self._prepared = False

        self._scaler = transforms.Scale((224, 224))
        # @TODO Magic numbers with no idea left of how they got here, we should at least have the decency of citing some paper that used these as magic numbers :P
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()

        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.ERROR)

    def to(self, device, available_devices):
        self.model.to(device, available_devices)
        return self

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if self.model is None:
            if self.aim == ENCODER_AIM.SPEED:
                self.model = Img2Vec(model='resnet-18')
            elif self.aim == ENCODER_AIM.BALANCE:
                self.model = Img2Vec(model='resnext-50-small')
            elif self.aim == ENCODER_AIM.ACCURACY:
                self.model = Img2Vec(model='resnext-50')
            else:
                self.model = Img2Vec()
        self._prepared = True

    def prepare(self, images):
        img_tensor_arr = []
        for image in images:
            if image is not None:
                if image.startswith('http'):
                    response = requests.get(image)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image)

                # In order for Normalize to work with 4-channel pngs and grayscale
                img = img.convert('RGB')

                img_tensor = self._scaler(img)
                img_tensor = self._to_tensor(img_tensor)
                img_tensor = self._normalize(img_tensor)
                img_tensor_arr.append(img_tensor)
            else:
                raise Exception('Can\'t work with images that are None')

        return torch.stack(img_tensor_arr)

    def encode(self, images):
        """
            Encode list of images

            :images : list of images, each image is a path to a file or a url
            :return: a torch.floatTensor
        """
        if not self._prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        img_tensors = self.prepare(images)
        vec_arr = []
        self.model.eval()
        with torch.no_grad():
            for img_tensor in img_tensors:
                vec = self.model(img_tensor.unsqueeze(0),batch=False)
                vec_arr.append(vec)
        return torch.stack(vec_arr)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')
