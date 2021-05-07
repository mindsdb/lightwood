import nltk
import re


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def contains_alnum(text):
    for c in text:
        if c.isalnum():
            return True
    return False


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def tokenize_text(text):
    return [t.lower() for t in nltk.word_tokenize(decontracted(text)) if contains_alnum(t)]
