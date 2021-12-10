import librosa
import torch
import warnings
from lightwood.encoder.base import BaseEncoder
from lightwood.helpers.io import read_from_path_or_url
from lightwood.helpers.log import log
from typing import Iterable


class MFCCEncoder(BaseEncoder):
    is_trainable_encoder: bool = False

    def __init__(self, is_target: bool = False):
        """
    
        Uses `librosa` to compute the Mel-frequency spectral coefficients (MFCCs) of data within an audio file. They are a common feature used in speech and audio processing. Example: https://centaur.reading.ac.uk/88046/3/ESR_for_home_AI.pdf

        Input data to this mixer MUST BE the location of the audio files.
    
        The output feature for any given audio file is a 2D array, flattened into a 1D one to comply with the expected format in lightwood mixers.
    
        This encoder currently does not support a `decode()` call; models with an audio output will not work. 
    
        :param is_target: whether this encoder's column is the target. Is always false, as decoded audio is not available.
        """  # noqa
        assert not is_target
        super().__init__(is_target)

    def prepare(self, priming_data: Iterable[str]):
        """
        Audio-encoding is rule-based; prepare only sets the size of the output dimension based on encoding the first example within the priming data.

        :param priming_data: Training data, in the form of file locations
        """  # noqa
        self.is_prepared = True
        priming_data = list(priming_data)
        ele = self.encode([str(priming_data[0])])[0]
        self.output_size = len(ele)

    def encode(self, column_data: Iterable[str]) -> torch.Tensor:
        """
        Encode a list of audio files via mfccs.

        :param column_data: list of strings that point to paths or URLs of the audio files that will be encoded.
        """
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
