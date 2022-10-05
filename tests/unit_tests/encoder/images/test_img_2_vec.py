import unittest

import torch
from torch import Tensor
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

    def run_test_encoder_on_device(self, device):
        enc = Img2VecEncoder(device=device)
        enc.prepare([], [])
        self.assertEqual(enc.model.device == torch.device(device))
        self.assertEqual(list(enc.model.parameters())[0].device.type == device)

    def test_encoder_on_cpu(self):
        self.run_test_encoder_on_device('cpu')

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA unavailable')
    def test_encoder_on_cuda(self):
        self.run_test_encoder_on_device('cuda')
