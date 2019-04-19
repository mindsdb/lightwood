from lightwood.column_data_types.text.helpers.rnn_helpers import *

class RnnEncoder:

    def __init__(self):
        self._input_lang = None
        self._output_lang = None
        self._encoder = None
        self._decoder = None
        self._trained = False

    def encode(self, column_data):

        if self._trained == False:
            self._input_lang = Lang('input')
            self._output_lang = self._input_lang

            for row in column_data:
                self._input_lang.addSentence(row)

            hidden_size = 256
            self._encoder = EncoderRNN(self._input_lang.n_words, hidden_size).to(device)
            self._decoder = DecoderRNN(hidden_size, self._output_lang.n_words, dropout_p=0.1).to(device)

            trainIters(self._encoder, self._decoder, self._input_lang, self._output_lang, column_data, column_data, 75000)

            self._trained = True


        ret = []
        with torch.no_grad():
            for row in column_data:

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
                ret+=[encoder_hidden.tolist()]

        return torch.FloatTensor(ret)


    def decode(self, encoded_values_tensor, max_length = 100):

        ret = []
        with torch.no_grad():
            for decoder_hidden in encoded_values_tensor:

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

                ret += [' '.join(decoded_words)]

        return ret