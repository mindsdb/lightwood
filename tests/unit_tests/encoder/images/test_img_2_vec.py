import unittest

from torch import Tensor
from torch import device as torch_device
from torch.cuda import is_available as torch_cuda_is_available
from lightwood.encoder.image import Img2VecEncoder
import os


class TestImg2VecEncoder(unittest.TestCase):
    def test_encode(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        image_path = os.path.join(
            dir_path, 'test_image.jpg'
        )

        enc = Img2VecEncoder()
        enc.prepare([], [])
        encoded_images_tensor = enc.encode(images=[image_path])

        self.assertTrue(isinstance(encoded_images_tensor, Tensor))

        # We expect the first dimension to equal the number of images
        self.assertEqual(encoded_images_tensor.size(0), 1)

        # We expect the second dimesion to be 512
        # NOTE: this will break when it will possible to choose different
        # encoding models.
        self.assertEqual(encoded_images_tensor.size(1), 512)

    def test_encoder_on_cpu(self):
        enc = Img2VecEncoder(device='cpu')
        enc.prepare([], [])
        self.assertEqual(enc.model.device, torch_device('cpu'))
        self.assertEqual(list(enc.model.parameters())[0].device.type, 'cpu')

    def test_encoder_on_cuda(self):
        if(not torch_cuda_is_available()):
            return # can't test if there is no Cuda GPU
        enc = Img2VecEncoder(device='cuda')
        enc.prepare([], [])
        assert(enc.model.device == torch_device('cuda'))
        assert(list(enc.model.parameters())[0].device.type == 'cuda')
