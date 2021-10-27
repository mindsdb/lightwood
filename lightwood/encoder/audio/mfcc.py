import librosa
import torch
import warnings
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.io import read_from_path_or_url
import pandas as pd
from lightwood.helpers.log import log


class MFCCEncoder(BaseEncoder):
    """
    Audio encoder. Uses `librosa` to compute the Mel-frequency spectral coefficients (MFCCs) of the audio file. They are a common feature used in speech and audio processing. The features are a 2D array, flattened into a 1D one.
    """  # noqa
    is_trainable_encoder: bool = False

    def __init__(self, is_target: bool = False):
        super().__init__(is_target)

    def prepare(self, priming_data: pd.Series):
        self.is_prepared = True
        priming_data = list(priming_data)
        ele = self.encode([str(priming_data[0])])[0]
        self.output_size = len(ele)

    def encode(self, column_data):
        encoded_audio_arr = []
        for path in column_data:
            try:
                y, _ = read_from_path_or_url(path, librosa.load)
            except Exception as e:
                log.error(f'Unable to read audio file {path}, error: {e}')
                encoded_audio_arr.append([0] * self.output_size)
                continue

            # If the durations of the audio samples are highly variable, the
            # same coefficients will refer to time buckets of different lenghts.
            # This means that a model will find difficult to use temporal
            # information
            NUM_TIME_BUCKETS = 100  # audio file will be split into 100 sequential time intervals before computing the Fourier transform needed for the MFCCs. A value of `100` will split a 1s audio file in 10ms intervals, which are enough for speech recognition. It will also split a 3 minutes song in 1.8s intervals, which are still small enough to capture enough detail for genre recognition  # noqa
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
