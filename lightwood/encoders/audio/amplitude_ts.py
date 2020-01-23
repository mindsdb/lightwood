from pydub import AudioSegment
import requests
import numpy as np

from lightwood.encoders.categorical.onehot import CesiumTsEncoder



class AmplitudeTsEncoder:

    def __init__(self, is_target = False):
        self._pytorch_wrapper = torch.FloatTensor
        self._ts_encoder = CesiumTsEncoder()

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, column_data):
        encoded_audio_arr = []
        for path in  column_data:
            if path.startswith('http'):
                response = requests.get(path)
                audio = AudioSegment.from_raw(BytesIO(response.content))
            else:
                audio = AudioSegment.from_file(path)
            # For now convert all (usually will be stereo) to mono by adding up and averging the amplitudes
            audio = audio.set_channels(1)
            audio_arr = np.array(audio.get_array_of_samples())
            encoded_audio_arr = self._ts_encoder.encode(audio_arr)
            encoded_audio_arr.append(encoded_audio_arr)

        return self._pytorch_wrapper(encoded_audio_arr)

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')

if __name__ == "__main__":
    encoder = AmplitudeTsEncoder()

    audio_url_arr = ['https://file-examples.com/wp-content/uploads/2017/11/file_example_MP3_1MG.mp3']

    encoded_audio = encoder.encode(audio_url_arr)
    print(encoded_audio)
