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


def timed_predictor(f):
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
        name_ = f.__name__ if f.__name__ != 'fit_mixer' else type(args[0]).__name__ + '.fit_mixer'
        if hasattr(predictor, 'runtime_log'):
            if name_ not in predictor.runtime_log:
                predictor.runtime_log[name_] = [(round(te - ts, 2), datetime.fromtimestamp(ts))]
            else:
                predictor.runtime_log[name_].append((round(te - ts, 2), datetime.fromtimestamp(ts)))
        return result
    return wrap


def timed(f):
    """
    Intended to be called from within any lightwood method to log the runtime.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        log.debug(f' `{f.__name__}` runtime: {round(te - ts, 2)} seconds')
        return result
    return wrap


log = initialize_log()
