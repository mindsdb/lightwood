# flake8: noqa
from lightwood.encoders.text.helpers.rnn_helpers import *
import logging
import math


class RnnEncoder:

    def __init__(self, encoded_vector_size=4, train_iters=75000, stop_on_error=0.0001,
                 learning_rate=0.01, is_target=False):
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._train_iters = train_iters

        self._encoder = None
        self._decoder = None
        self._pytorch_wrapper = torch.FloatTensor
        self._prepared = False

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        no_null_sentences = [x if x is not None else [] for x in priming_data]
        estimated_time = 1/937*self._train_iters*len(no_null_sentences)
        log_every = math.ceil(self._train_iters/100)
        logging.info('We will train an encoder for this sequence, on a CPU it will take about {min} minutes'.format(
            min=estimated_time))


        max_length = max(map(len, no_null_sentences))

        hidden_size = self._encoded_vector_size
        self._encoder = EncoderRNNNumerical(1, hidden_size).to(device)
        self._decoder = DecoderRNNNumerical(hidden_size, 1).to(device)

        trainItersNoLang(self._encoder, self._decoder, no_null_sentences, no_null_sentences, self._train_iters, int(log_every), self._learning_rate, self._stop_on_error,
                   max_length)

        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        no_null_sentences = [x if x is not None else [] for x in column_data]
        ret = []
        with torch.no_grad():
            for row in no_null_sentences:

                encoder_hidden = self._encoder.initHidden()
                input_tensor = tensorFromSeries(row)
                input_length = input_tensor.size(0)

                #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                loss = 0

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self._encoder(
                        input_tensor[ei], encoder_hidden)
                    #encoder_outputs[ei] = encoder_output[0, 0]

                # use the last hidden state as the encoded vector
                ret += [encoder_hidden.tolist()[0][0]]

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor, max_length=100):

        ret = []
        with torch.no_grad():
            for decoder_hidden in encoded_values_tensor:
                decoder_hidden = torch.tensor([[decoder_hidden.tolist()]], device=device).float()
                decoder_input = torch.tensor([[[SOS_token]]], device=device)  # SOS

                decoded_words = []

                for di in range(max_length):
                    decoder_output, decoder_hidden = self._decoder(
                        decoder_input, decoder_hidden)

                    topv, topi = decoder_output.data.topk(1)
                    decoded_words.append( topv.tolist()[0][0][0] )


                    decoder_input = topv.detach()

                ret += [decoded_words]

        return ret


# only run the test if this file is called from debugger
if __name__ == "__main__":
    series = [[10,20,30,40,50],
                 [1,2,3,4,5,6,7],
                 [5,7,9,11,13,15,17,19],
                 [3,6,9,12,15,18,21,24,27,30]

                 ]

    encoder = RnnEncoder(encoded_vector_size=3,train_iters=75000)
    encoder.prepare_encoder(series)
    encoder.encode(series)

    # test de decoder

    ret = encoder.encode([[3,4,5,6,7]])
    print('encoded vector:')
    print(ret)
    print('decoded vector')
    ret2 = encoder.decode(ret)
    print(ret2)
