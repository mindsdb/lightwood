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
import json
import hashlib
import nltk


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)


def word_tokenize(string):
    sep_tag = '{#SEP#}'
    for separator in WORD_SEPARATORS:
        string = str(string).replace(separator, sep_tag)

    words_split = string.split(sep_tag)
    num_words = len([word for word in words_split if word and word not in ['', None]])
    return num_words


def gen_chars(length, character):
    """
    # lambda to Generates a string consisting of `length` consiting of repeating `character`
    :param length:
    :param character:
    :return:
    """
    return ''.join([character for _ in range(length)])


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
