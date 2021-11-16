import unittest
import random
import torch
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score
from lightwood.encoder import BinaryEncoder
from lightwood.encoder.text import PretrainedLangEncoder
from lightwood.api.dtype import dtype
import pandas as pd

from nltk.corpus import opinion_lexicon


def create_fake_language(n, ptrain=0.7, seed=2):
    """
    Returns "N" instances of a fake language.

    Labels are 0/1; 0 -> negative sentiment, 1-> positive sentiment
    
    Creates a nonsense string of positive/negative words with some random subset of them.

    :param n: the maximum character (n-1) in the string
    """
    # Pick positive/negative words
    pos_list=list(opinion_lexicon.positive())
    neg_list=list(opinion_lexicon.negative())

    random.seed(seed)
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

    Ntrain = int(n*ptrain)
    train = pd.DataFrame([data[:Ntrain], label[:Ntrain]]).T
    test = pd.DataFrame([data[Ntrain:], label[Ntrain:]]).T

    train.columns = ["text", "label"]
    test.columns = ["text", "label"]
    return train, test


class TestPretrainedLangEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        """
        Test end-to-end training of values. Performance metric doesn't matter,
        just to check if it compiles properly
        """
        seed = 2

        # Make priming data:
        train, test = create_fake_language(1000)
        random.seed(seed)
        output_enc = BinaryEncoder(is_target=True)
        output_enc.prepare(train["label"])
        encoded_target_values = output_enc.encode(train["label"])

        # Prepare the language encoder
        enc = PretrainedLangEncoder(stop_after=10, embed_mode=False, output_type='binary')
        enc.prepare(train["text"], None, encoded_target_values=encoded_target_values)


        test_labels = test["label"].tolist()
        pred_labels = softmax(enc.encode(test["text"]),dim=1).argmax(dim=1).tolist()

        encoder_accuracy = accuracy_score(test_labels, pred_labels)

        print(f'Categorial encoder accuracy for: {encoder_accuracy} on testing dataset')
            # assert(encoder_accuracy > 0.5)
