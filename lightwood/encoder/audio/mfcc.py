import os
import librosa
import requests
import numpy as np
import torch
import warnings
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.log import log


class MFCCEncoder(BaseEncoder):
    def __init__(self, is_target: bool = False):
        super().__init__(is_target)

    def encode(self, column_data):
        encoded_audio_arr = []
        for path in column_data:
            if path.startswith('http'):
                response = requests.get(path)
                with open(path.split('/')[-1], 'wb') as f:
                    f.write(response.content)
                try:
                    y, sr = librosa.load(path.split('/')[-1])
                except Exception as e:
                    log.error(e)
                finally:
                    os.remove(path.split('/')[-1])
            else:
                # Will automatically resample to 22.05kHz and convert to mono
                y, _ = librosa.load(path)

            # If the durations of the audio samples are highly variable, the
            # same coefficients will refer to time buckets of different lenghts.
            # This means that a model will find difficult to use temporal
            # information
            NUM_TIME_BUCKETS = 100
            N_MFCC_COEFFICIENTS = 20

            num_samples = y.shape[0]

            # truncate towards 1
            hop_length = int(num_samples / NUM_TIME_BUCKETS + 1)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                mfcc_coefficients = librosa.feature.mfcc(
                    y,
                    n_mfcc=N_MFCC_COEFFICIENTS,
                    hop_length=hop_length
                ).reshape(-1)

            encoded_audio_arr.append(mfcc_coefficients)

        return torch.Tensor(encoded_audio_arr)

    def decode(self, _):
        raise Exception('This encoder is not bi-directional')
