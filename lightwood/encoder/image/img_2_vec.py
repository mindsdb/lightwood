from typing import List
import torch
import pandas as pd
from lightwood.encoder.image.helpers.img_to_vec import Img2Vec
from lightwood.encoder.base import BaseEncoder

from lightwood.helpers.log import log

try:
    import torchvision.transforms as transforms
    from PIL import Image
except ModuleNotFoundError:
    log.info("No torchvision/pillow detected, image encoder not supported")


class Img2VecEncoder(BaseEncoder):
    """
    This encoder generates encoded representations for images using a pre-trained deep neural network.

    All input images are rescaled to a standard size of 224x224, and normalized using the mean and standard deviation of the ImageNet dataset (as it was used to train the underlying NN).
    
    Note that this encoder does not have a .decode() method yet. As such, models that predict images as output are not supported at this time. 
    
    For more information about the neural network this encoder uses, refer to the `lightwood.encoder.image.helpers.img_to_vec.Img2Vec`.
    """  # noqa

    is_trainable_encoder: bool = True

    def __init__(self, stop_after: float = 3600, is_target: bool = False):
        """
        :param stop_after: time budget, in seconds. 
        :param is_target: whether the encoder corresponds to the target column. This is not currently possible for Img2VecEncoder.
        """  # noqa
        assert not is_target
        super().__init__(is_target)
        self.is_prepared = False

        self._scaler = transforms.Resize((224, 224))
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._to_tensor = transforms.ToTensor()
        self._img_to_tensor = transforms.Compose([
            self._scaler,
            self._to_tensor,
            self._normalize
        ])
        self.stop_after = stop_after

        #pil_logger = logging.getLogger('PIL')
        #pil_logger.setLevel(logging.ERROR)

    def prepare(self, train_priming_data: pd.Series, dev_priming_data: pd.Series):
        # @TODO: finetune here? depending on time aim
        """
        Instances an `Img2Vec` object and sets the expected size for encoded representations.
        """
        if self.is_prepared:
            raise Exception('You can only call "prepare" once for a given encoder.')

        self.model = Img2Vec()
        self.output_size = self.model.output_size
        self.is_prepared = True

    def to(self, device, available_devices):
        """
        Moves the model to-and-from CPU and GPU.

        :param device: will move the model to this device.
        :param available_devices: all available devices as reported by lightwood.

        :return: same object but moved to the target device.
        """
        self.model.to(device, available_devices)
        return self

    def encode(self, images: List[str]) -> torch.Tensor:
        """
        Creates encodings for a list of images; each image is referenced by a filepath or url.

        :param images: list of images, each image is a path to a file or a url.
        :return: a torch.floatTensor
        """
        if not self.is_prepared:
            raise Exception('You need to call "prepare" before calling "encode" or "decode".')

        img_tensors = [self._img_to_tensor(
            Image.open(img_path)
        ) for img_path in images]
        vec_arr = []

        self.model.eval()
        with torch.no_grad():
            for img_tensor in img_tensors:
                vec = self.model(img_tensor.unsqueeze(0), batch=False)
                vec_arr.append(vec)
        return torch.stack(vec_arr).to('cpu')

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')
