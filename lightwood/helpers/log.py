import colorlog
import logging
import os

def initialize_log():
    pid = os.getpid()
    logging.basicConfig()
    log = logging.getLogger(f'lightwood-{pid}')

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s', log_colors={
        'WARNING': 'red',
        'ERROR': 'red,bg_white',
        'CRITICAL': 'red,bg_white',
    }))
    log.addHandler(handler)
    
    log.setLevel(logging.DEBUG)
    return log


log = initialize_log()
