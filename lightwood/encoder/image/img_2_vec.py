import logging
import torch
import torchvision.transforms as transforms
from lightwood.encoder.image.helpers.img_to_vec import Img2Vec
from lightwood.encoder.base import BaseEncoder


class Img2VecEncoder(BaseEncoder):

    def __init__(self, is_target: bool = False):
        super().__init__(is_target)
        self.model = None
        # I think we should make this an enum, something like: speed, balance, accuracy
        self.aim = aim
        self._prepared = False

        self._scaler = transforms.Scale((224, 224))
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()

        pil_logger = logging.getLogger('PIL')
        pil_logger.setLevel(logging.ERROR)

    def prepare(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        if self.model is None:
            self.model = Img2Vec(model='resnext-50-small')
        self._prepared = True

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
                vec = self.model(img_tensor.unsqueeze(0), batch=False)
                vec_arr.append(vec)
        return torch.stack(vec_arr)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')
