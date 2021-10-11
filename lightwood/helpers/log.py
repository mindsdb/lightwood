import logging
import os
import colorlog


def initialize_log():
    pid = os.getpid()

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter())

    logging.basicConfig(handlers=[handler])
    log = logging.getLogger(f'lightwood-{pid}')
    log.setLevel(logging.DEBUG)
    return log


log = initialize_log()
