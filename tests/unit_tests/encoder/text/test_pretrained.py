import unittest
import numpy as np
import random
import torch
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
from lightwood.encoder import BinaryEncoder, NumericEncoder
from lightwood.encoder.text import PretrainedLangEncoder
from lightwood.api.dtype import dtype
import pandas as pd
import os
import pathlib


def create_synthetic_data(n, ptrain=0.7):
    """
    Returns "N" instances of a fake language.

    Labels are 0/1; 0 -> negative sentiment, 1-> positive sentiment
    
    Creates a nonsense string of positive/negative words with some random subset of them.

    :param n: the maximum character (n-1) in the string
    """ # noqa

    textdir = str(pathlib.Path(__file__).parent.resolve())

    # Pick positive/negative words
    with open(os.path.join(textdir, "pos.txt"), "r") as f:
        pos_list = f.readlines()
        pos_list = [i.strip("\n") for i in pos_list]

    with open(os.path.join(textdir, "neg.txt"), "r") as f:
        neg_list = f.readlines()
        neg_list = [i.strip("\n") for i in neg_list]

    data = []
    label = []

    for i in range(n):
        y = random.randint(0, 1)

        # Negative words
        if y == 0:
            word = random.choice(neg_list)
        # Positive words
        else:
            word = random.choice(pos_list)

        data.append(word)
        label.append(y)

    Ntrain = int(n * ptrain)
    train = pd.DataFrame([data[:Ntrain], label[:Ntrain]]).T
    test = pd.DataFrame([data[Ntrain:], label[Ntrain:]]).T

    train.columns = ["text", "label"]
    test.columns = ["text", "label"]
    return train, test


class TestPretrainedLangEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        """
        Test end-to-end training of values. 

        Uses labeled data with positive/negative sentiment. Model outputs logits, since embed_mode is false. Should be better than random, as transformer language models may actually capture semantics (This is a hypothesis from priming lit.)
        """ # noqa
        seed = 2

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Make priming data:
        train, test = create_synthetic_data(1000)
        output_enc = BinaryEncoder(is_target=True)
        output_enc.prepare(train["label"])
        encoded_target_values = output_enc.encode(train["label"])

        # Prepare the language encoder
        enc = PretrainedLangEncoder(stop_after=10, embed_mode=False, output_type=dtype.binary)
        enc.prepare(train["text"], pd.DataFrame(), encoded_target_values=encoded_target_values)

        test_labels = test["label"].tolist()
        pred_labels = softmax(enc.encode(test["text"]), dim=1).argmax(dim=1).tolist()

        encoder_accuracy = accuracy_score(test_labels, pred_labels)

        # Should be non-random since models have primed associations to sentiment
        print(f'Categorial encoder accuracy for: {encoder_accuracy} on testing dataset')

    def test_embed_mode(self):
        """
        Test if embed-mode is triggered when flagged.
        Checks if returned embeddings are of size N_rows x N_embed_dim
        """ # noqa
        seed = 2

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Make priming data:
        train, test = create_synthetic_data(1000)
        output_enc = BinaryEncoder(is_target=True)
        output_enc.prepare(train["label"])
        encoded_target_values = output_enc.encode(train["label"])

        # Prepare the language encoder
        enc = PretrainedLangEncoder(stop_after=10, embed_mode=True, output_type=dtype.binary)
        enc.prepare(train["text"], pd.DataFrame(), encoded_target_values=encoded_target_values)

        # Embeddings of size N_vocab x N_embed_dim for most models (assumes distilbert)
        N_embed_dim = enc._model.base_model.embeddings.word_embeddings.weight.shape[-1]
        embeddings = enc.encode(test["text"])
        assert(embeddings.shape[0] == test.shape[0])
        assert(embeddings.shape[1] == N_embed_dim)

    def test_auto_embed_mode(self):
        """
        For regression (non-categorical type output), text defaults to embed mode as fine-tuning didn't seem to help. Check to see if the output is in fact the size of an embedding. 

        We pretend "embed_mode" is not True, but it should auto-override.

        Expected behavior:
        - does not train the transformer
        - flips transformer to 'embed_mode'
        - returns embedding size N_rows x N_embed_dim
        """ # noqa
        seed = 5

        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Make priming data:
        train, test = create_synthetic_data(2000)
        output_enc = NumericEncoder(is_target=True)
        output_enc.prepare(train["label"])
        encoded_target_values = output_enc.encode(train["label"])

        # Prepare the language encoder
        enc = PretrainedLangEncoder(stop_after=10, embed_mode=False, output_type=dtype.float)
        enc.prepare(train["text"], pd.DataFrame(), encoded_target_values=encoded_target_values)

        # Embeddings of size N_vocab x N_embed_dim for most models (assumes distilbert)
        N_embed_dim = enc._model.base_model.embeddings.word_embeddings.weight.shape[-1]
        embeddings = enc.encode(test["text"])
        assert(enc.embed_mode)
        assert(embeddings.shape[0] == test.shape[0])
        assert(embeddings.shape[1] == N_embed_dim)


