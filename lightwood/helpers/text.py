"""
*******************************************************
 * Copyright (C) 2017 MindsDB Inc. <copyright@mindsdb.com>
 *
 * This file is part of MindsDB Server.
 *
 * MindsDB Server can not be copied and/or distributed without the express
 * permission of MindsDB Inc
 *******************************************************
"""
from collections import Counter, defaultdict
import string
import json
import re
import hashlib
from typing import Iterable
import numpy as np
import scipy.stats as st
import langdetect
import nltk
from lightwood.api.dtype import dtype


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


def get_language_dist(data):
    langdetect.DetectorFactory.seed = 0
    lang_dist = defaultdict(lambda: 0)
    lang_dist['Unknown'] = 0
    lang_probs_cache = dict()
    for text in data:
        text = str(text)
        text = ''.join([c for c in text if c not in string.punctuation])
        if text not in lang_probs_cache:
            try:
                lang_probs = langdetect.detect_langs(text)
            except langdetect.lang_detect_exception.LangDetectException:
                lang_probs = []
            lang_probs_cache[text] = lang_probs

        lang_probs = lang_probs_cache[text]
        if len(lang_probs) > 0 and lang_probs[0].prob > 0.90:
            lang_dist[lang_probs[0].lang] += 1
        else:
            lang_dist['Unknown'] += 1

    return dict(lang_dist)


def analyze_sentences(data):
    """
    :param data: list of str

    :returns:
    tuple(
        int: nr words total,
        dict: word_dist,
        dict: nr_words_dist
    )
    """
    nr_words = 0
    word_dist = defaultdict(int)
    nr_words_dist = defaultdict(int)
    stop_words = set(stopwords.words('english'))
    for text in map(str, data):
        text = text.lower()
        tokens = tokenize_text(text)
        tokens_no_stop = [x for x in tokens if x not in stop_words]
        nr_words_dist[len(tokens)] += 1
        nr_words += len(tokens)
        for tok in tokens_no_stop:
            word_dist[tok] += 1

    return nr_words, dict(word_dist), dict(nr_words_dist)


def word_tokenize(string):
    sep_tag = '{#SEP#}'
    for separator in WORD_SEPARATORS:
        string = str(string).replace(separator, sep_tag)

    words_split = string.split(sep_tag)
    num_words = len([word for word in words_split if word and word not in ['', None]])
    return num_words


def clean_float(val):
    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, float):
        return val

    val = str(val).strip(' ')
    val = val.replace(',', '.')
    val = val.rstrip('"').lstrip('"')

    if val in ('', '.', 'None', 'nan'):
        return None

    return float(val)


def gen_chars(length, character):
    """
    # lambda to Generates a string consisting of `length` consiting of repeating `character`
    :param length:
    :param character:
    :return:
    """
    return ''.join([character for _ in range(length)])


def cast_string_to_python_type(string):
    """ Returns None, an integer, float or a string from a string"""
    if string is None or string == '':
        return None

    if string.isnumeric():
        # Did you know you can write fractions in unicode, and they are numeric but can't be cast to integers !?
        try:
            return int(string)
        except Exception:
            return None

    try:
        return clean_float(string)
    except Exception:
        return string


def splitRecursive(word, tokens):
    words = [str(word)]
    for token in tokens:
        new_split = []
        for word in words:
            new_split += word.split(token)
        words = new_split
    words = [word for word in words if word not in ['', None]]
    return words


def hashtext(cell):
    text = json.dumps(cell)
    return hashlib.md5(text.encode('utf8')).hexdigest()


def _is_foreign_key_name(name):
    for endings in ['id', 'ID', 'Id']:
        for add in ['-', '_', ' ']:
            if name.endswith(add + endings):
                return True
    for endings in ['ID', 'Id']:
        if name.endswith(endings):
            return True
    return False


def _is_identifier_name(name):
    for keyword in ['account', 'uuid', 'identifier', 'user']:
        if keyword in name:
            return True
    return False


def isascii(string):
    """
    Used instead of str.isascii because python 3.6 doesn't have that
    """
    return all(ord(c) < 128 for c in string)


def extract_digits(point):
    return ''.join([char for char in str(point) if char.isdigit()])


def get_pct_auto_increment(data):
    int_data = []
    for point in [extract_digits(x) for x in data]:
        try:
            int_data.append(int(point))
        except Exception:
            pass

    int_data = sorted(int_data)
    prev_nr = int_data[0]
    increase_by_one = 0
    for nr in int_data[1:]:
        diff = nr - prev_nr
        if diff == 1:
            increase_by_one += 1
        prev_nr = nr

    return increase_by_one / (len(data) - 1)


def get_identifier_description_mp(arg_tup):
    data, column_name, data_dtype = arg_tup
    return get_identifier_description(data, column_name, data_dtype)


def get_identifier_description(data: Iterable, column_name: str, data_dtype: dtype):
    data = list(data)
    if isinstance(data[0], list):
        nr_unique = len(set(tuple(x) for x in data))
    else:
        nr_unique = len(set(data))

    if nr_unique == 1:
        return 'No Information'

    unique_pct = nr_unique / len(data)

    spaces = [len(str(x).split(' ')) - 1 for x in data]
    mean_spaces = np.mean(spaces)

    # Detect auto incrementing index
    # -- some cases where I guess people do want to use this for learning, so ignoring this check for now...
    # if data_dtype == dtype.integer:
    #    if get_pct_auto_increment(data) > 0.98 and unique_pct > 0.99:
    #        return 'Auto-incrementing identifier'

    # Detect hash
    all_same_length = all(len(str(data[0])) == len(str(x)) for x in data)
    uuid_charset = set('0123456789abcdefABCDEF-')
    all_uuid_charset = all(set(str(x)).issubset(uuid_charset) for x in data)
    is_uuid = all_uuid_charset and all_same_length

    if all_same_length and len(data) == nr_unique and data_dtype not in (dtype.integer, dtype.float):
        str_data = [str(x) for x in data]
        randomness_per_index = []
        for i, _ in enumerate(str_data[0]):
            N = len(set(x[i] for x in str_data))
            S = st.entropy([*Counter(x[i] for x in str_data).values()])
            randomness_per_index.append(S / np.log(N))

        if np.mean(randomness_per_index) > 0.95:
            return 'Hash-like identifier'

    # Detect foreign key
    if data_dtype == dtype.integer:
        if _is_foreign_key_name(column_name):
            return 'Foreign key'

    if _is_identifier_name(column_name) or data_dtype in (dtype.categorical, dtype.binary):
        if unique_pct > 0.98:
            if is_uuid:
                return 'UUID'
            else:
                return 'Unknown identifier'

    # Everything is unique and it's too short to be rich text
    if data_dtype in (dtype.categorical, dtype.binary, dtype.short_text, dtype.rich_text) and \
            unique_pct > 0.99999 and mean_spaces < 1:
        return 'Unknown identifier'

    return None


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
