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
