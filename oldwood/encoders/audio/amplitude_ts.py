import os

from pydub import AudioSegment
import requests
import numpy as np
import torch

from lightwood.encoders.time_series.rnn import RnnEncoder
from lightwood.encoders.encoder_base import BaseEncoder
from lightwood.logger import log


class AmplitudeTsEncoder(BaseEncoder):

    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._ts_encoder = RnnEncoder()
        self._max_samples = 2000

    def encode(self, column_data):
        encoded_audio_arr = []
        for path in  column_data:
            if path.startswith('http'):
                response = requests.get(path)
                with open(path.split('/')[-1], 'wb') as f:
                    f.write(response.content)
                try:
                    audio = AudioSegment.from_file(path.split('/')[-1])
                except Exception as e:
                    print(e)
                finally:
                    os.remove(path.split('/')[-1])
            else:
                audio = AudioSegment.from_file(path)
            # For now convert all (usually will be stereo) to mono by adding up and averging the amplitudes
            audio = audio.set_channels(1)

            original_frame_rate = audio.frame_rate
            new_frame_rate = int(original_frame_rate/(len(audio.get_array_of_samples())/self._max_samples))

            if new_frame_rate < original_frame_rate:
                audio = audio.set_frame_rate(new_frame_rate)
                log.info(f'Lowering audio frame rate from {original_frame_rate} to {new_frame_rate} for ease of processing !')

            audio_arr = list(np.array(audio.get_array_of_samples()))

            encoded_audio = self._ts_encoder.encode([audio_arr])

            encoded_audio_arr.append(encoded_audio[0])

        return torch.Tensor(encoded_audio_arr)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')
