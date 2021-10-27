import os
import unittest

from torch import Tensor

from lightwood.encoder.audio import MFCCEncoder


class TestMFCCEncoder(unittest.TestCase):
    def test_encode(self):
        if MFCCEncoder is None:
            print('Skipping this test since the system for the encoder work are not installed')
            return

        dir_path = os.path.dirname(os.path.realpath(__file__))
        audio_paths = [
            os.path.join(dir_path, 'test_audio_1.wav'),
            os.path.join(dir_path, 'test_audio_2.wav'),
            None
        ]

        encoder = MFCCEncoder()
        encoder.prepare(audio_paths)
        encoded_audio = encoder.encode(audio_paths)

        self.assertTrue(isinstance(encoded_audio, Tensor))
        # We expect the first dimension to equal the number of images
        self.assertEqual(encoded_audio.size(0), 3)
        self.assertEqual(encoded_audio.size(1), 2000)
