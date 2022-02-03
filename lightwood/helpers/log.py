import os
import logging
import colorlog
from time import time
from datetime import datetime
from functools import wraps


def initialize_log():
    pid = os.getpid()

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'lightwood-{pid}')
    log_level = os.environ.get('LIGHTWOOD_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


def timed(f):
    """
    Intended to be called from within lightwood predictor methods.
    We use `wraps` to pass metadata into debuggers (as in stackoverflow.com/a/27737385)
    """
    @wraps(f)
    def wrap(predictor, *args, **kw):
        ts = time()
        result = f(predictor, *args, **kw)
        te = time()
        log.debug(f' `{f.__name__}` runtime: {round(te - ts, 2)} seconds')
        predictor.runtime_log[(f.__name__, datetime.fromtimestamp(ts))] = te - ts
        return result
    return wrap


log = initialize_log()
