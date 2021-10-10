import logging
import os


def initialize_log():
    pid = os.getpid()
    logging.basicConfig()
    log = logging.getLogger(f'lightwood-{pid}')
    log_level = os.environ.get('LIGHTWOOD_LOG', 'DEBUG')
    log.setLevel(log_level)
    return log


log = initialize_log()
