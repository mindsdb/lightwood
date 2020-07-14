import os
import shutil
import sys
import tempfile
import zipfile
from urllib.parse import urlparse
from urllib.request import urlopen

import nltk
import torch
from torch.utils.model_zoo import tqdm

import logging

from lightwood.encoders.text.helpers.infersent import InferSent
from lightwood.encoders.encoder_base import BaseEncoder

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

PARAMS_MODEL = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
MODEL_PATH = "pkl_objects/infersent2.pkl"
W2V_PATH = "datasets/fastText/crawl-300d-2M-subword.vec"


class InferSentEncoder(BaseEncoder):

    def __init__(self, is_target=False):
        super().__init__(is_target)
        self._model = None

    def prepare_encoder(self, priming_data):
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        self._download_necessary_files()

        no_null_sentences = [x if x is not None else '' for x in priming_data]

        if self._model is None:
            self._model = InferSent(PARAMS_MODEL)
            self._model.load_state_dict(torch.load(MODEL_PATH))
            self._model.set_w2v_path(W2V_PATH)

            self._model.build_vocab(no_null_sentences, tokenize=True)

        self._prepared = True

    def encode(self, sentences):
        """
        Encode a column of sentences

        :param sentences: a list of sentences
        :return: a torch.floatTensor
        """
        if not self._prepared:
            raise Exception('You need to call "prepare_encoder" before calling "encode" or "decode".')

        no_null_sentences = [x if x is not None else '' for x in sentences]
        result = self._model.encode(no_null_sentences, bsize=128, tokenize=False, verbose=True)
        return self._pytorch_wrapper(result)

    def _download_necessary_files(self):
        self._download_model_file()
        self._download_embeddings_file()

    def _download_model_file(self):
        pkl_dir = "pkl_objects/"
        pkl_url = "https://dl.fbaipublicfiles.com/infersent/infersent2.pkl"
        if not os.path.exists(pkl_dir):
            os.makedirs(pkl_dir)
        if not os.path.exists(MODEL_PATH):
            logging.info('This is the first time you use this text encoder, we will download a pretrained model.')
            sys.stderr.write('Downloading: "{}" to {}\n'.format(pkl_url, MODEL_PATH))
            self._download_url_to_file(pkl_url, MODEL_PATH, progress=True)

    def _download_embeddings_file(self):
        emdeddings_dir = "datasets/fastText/"
        embeddings_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
        if not os.path.exists(emdeddings_dir):
            os.makedirs(emdeddings_dir)
        parts = urlparse(embeddings_url)
        filename = os.path.basename(parts.path)
        cached_zip_file = os.path.join(emdeddings_dir, filename)
        if not os.path.exists(W2V_PATH):
            logging.info('We will download word embeddings, this will take about 20 minutes.')
            sys.stderr.write('Downloading: "{}" to {}\n'.format(embeddings_url, cached_zip_file))
            self._download_url_to_file(embeddings_url, cached_zip_file, progress=True)
            self._unzip_file(cached_zip_file, emdeddings_dir)
            os.remove(cached_zip_file)
            os.remove(cached_zip_file.replace("zip", "bin"))

    def _unzip_file(self, path, file_dir):
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(file_dir)

    def _download_url_to_file(self, url, dst, progress):
        file_size = None
        u = urlopen(url)
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            content_length = meta.getheaders("Content-Length")
        else:
            content_length = meta.get_all("Content-Length")
        if content_length is not None and len(content_length) > 0:
            file_size = int(content_length[0])

        f = tempfile.NamedTemporaryFile(delete=False)
        try:
            with tqdm(total=file_size, disable=not progress) as pbar:
                while True:
                    buffer = u.read(8192)
                    if len(buffer) == 0:
                        break
                    f.write(buffer)
                    pbar.update(len(buffer))

            f.close()
            shutil.move(f.name, dst)
        finally:
            f.close()
            if os.path.exists(f.name):
                os.remove(f.name)


# only run the test if this file is called from debugger
if __name__ == "__main__":
    #TODO: check _download_embeddings_file, it will download files after each run which takes 5-10min
    sentences = ["Everyone really likes the newest benefits",
                 "The Government Executive articles housed on the website are not able to be searched",
                 "Most of Mrinal Sen 's work can be found in European collections . ",
                 "Would you rise up and defeaat all evil lords in the town ? ",
                 None
                 ]

    encoder = InferSentEncoder()
    encoder.prepare_encoder(sentences)
    ret = encoder.encode(sentences)
    print(ret)

    ret = encoder.encode(["And they will fail to raise"])
    print(ret)

    ret = encoder.encode(["Everyone really likes the newest benefits"])
    print(ret)
