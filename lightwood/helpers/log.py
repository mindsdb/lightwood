import os
import logging


def initialize_log():
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    return log


log = initialize_log()
