import torch

from lightwood.column_data_types.text.helpers.infersent import InferSent
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class InferSentEncoder:

    def __init__(self, model_version=2):
        self._model_version = model_version
        self.MODEL_PATH = "../../../../pkl_objects/infersent%s.pickle" % self._model_version
        self.W2V_PATH = 'datasets/GloVe/glove.840B.300d.txt' if self._model_version == 1 else 'datasets/fastText/crawl-300d-2M-subword.vec'

    def encode(self, sentences):
        """
        Encode a column of sentences

        :param sentences: a list of sentences
        :return: a torch.floatTensor
        """

        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': self._model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(self.MODEL_PATH))

        model.set_w2v_path(self.W2V_PATH)

        model.build_vocab(sentences, tokenize=True)
        result = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
        ret_tensor = torch.FloatTensor(result)
        return ret_tensor


# only run the test if this file is called from debugger
if __name__ == "__main__":
    sentences = ["Everyone really likes the newest benefits",
                 "The Government Executive articles housed on the website are not able to be searched",
                 "Most of Mrinal Sen 's work can be found in European collections . ",
                 "Would you rise up and defeaat all evil lords in the town ? "
                 ]

    ret = InferSentEncoder(2).encode(sentences)

    print(ret)
