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
import os

import nltk
from contextlib import redirect_stdout

from lightwood.helpers.log import log


try:
    with redirect_stdout(open(os.devnull, "w")):
        nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception:
        log.error("NLTK was unable to download the 'punkt' package. Please check your connection and try again!")

try:
    with redirect_stdout(open(os.devnull, "w")):
        from nltk.corpus import stopwords
        stopwords.words('english')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except Exception:
        log.error("NLTK was unable to download the 'stopwords' package. Please check your connection and try again!")
