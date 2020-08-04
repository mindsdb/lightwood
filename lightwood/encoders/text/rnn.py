# flake8: noqa
from lightwood.encoders.text.helpers.rnn_helpers import *
from lightwood.encoders.encoder_base import BaseEncoder
import logging
import math


class RnnEncoder(BaseEncoder):

    def __init__(self, encoded_vector_size=256, train_iters=75000, stop_on_error=0.0001,
                 learning_rate=0.01, is_target=False):
        super().__init__(is_target)
        self._stop_on_error = stop_on_error
        self._learning_rate = learning_rate
        self._encoded_vector_size = encoded_vector_size
        self._train_iters = train_iters
        self._input_lang = None
        self._output_lang = None
        self._encoder = None
        self._decoder = None

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        no_null_sentences = [x if x is not None else '' for x in priming_data]
        estimated_time = 1/937*self._train_iters*len(no_null_sentences)
        log_every = math.ceil(self._train_iters/100)
        logging.info('We will train an encoder for this text, on a CPU it will take about {min} minutes'.format(
            min=estimated_time))

        self._input_lang = Lang('input')
        self._output_lang = self._input_lang

        for row in no_null_sentences:
            if row is not None:
                self._input_lang.addSentence(row)

        max_length = max(map(len, no_null_sentences))

        hidden_size = self._encoded_vector_size
        self._encoder = EncoderRNN(self._input_lang.n_words, hidden_size).to(device)
        self._decoder = DecoderRNN(hidden_size, self._output_lang.n_words).to(device)

        trainIters(self._encoder, self._decoder, self._input_lang, self._output_lang, no_null_sentences, no_null_sentences, self._train_iters, int(log_every), self._learning_rate, self._stop_on_error,
                   max_length)

        self._prepared = True

    def encode(self, column_data):
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        no_null_sentences = [x if x is not None else '' for x in column_data]
        ret = []
        with torch.no_grad():
            for row in no_null_sentences:

                encoder_hidden = self._encoder.initHidden()
                input_tensor = tensorFromSentence(self._input_lang, row)
                input_length = input_tensor.size(0)

                #encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                loss = 0

                for ei in range(input_length):
                    encoder_output, encoder_hidden = self._encoder(
                        input_tensor[ei], encoder_hidden)
                    #encoder_outputs[ei] = encoder_output[0, 0]

                # use the last hidden state as the encoded vector
                ret.append(encoder_hidden.tolist()[0][0])

        return self._pytorch_wrapper(ret)

    def decode(self, encoded_values_tensor, max_length=100):

        ret = []
        with torch.no_grad():
            for decoder_hiddens in encoded_values_tensor:
                decoder_hidden = torch.FloatTensor([[decoder_hiddens.tolist()]])

                decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

                decoded_words = []

                for di in range(max_length):
                    decoder_output, decoder_hidden = self._decoder(
                        decoder_input, decoder_hidden)

                    topv, topi = decoder_output.data.topk(1)
                    if topi.item() == EOS_token:
                        decoded_words.append('<EOS>')
                        break
                    else:
                        decoded_words.append(self._output_lang.index2word[topi.item()])

                    decoder_input = topi.squeeze().detach()

                ret.append(' '.join(decoded_words))

        return ret


# only run the test if this file is called from debugger
if __name__ == "__main__":
    sentences = ["Everyone really likes the newest benefits",
                 "The Government Executive articles housed on the website are not able to be searched",
                 "Most of Mrinal Sen 's work can be found in European collections . ",
                 "Would you rise up and defeaat all evil lords in the town ? ",
                 None

                 ]

    encoder = RnnEncoder(encoded_vector_size=10,train_iters=7500)
    encoder.prepare_encoder(sentences)
    encoder.encode(sentences)

    # test de decoder

    ret = encoder.encode(["Everyone really likes the newest benefits"])
    print('encoded vector:')
    print(ret)
    print('decoded vector')
    ret2 = encoder.decode(ret)
    print(ret2)
