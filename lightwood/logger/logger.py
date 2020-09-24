import os
import logging

log = logging.getLogger(f'lightwood.{os.getpid()}')
log_level = os.environ.get('LIGHTWOOD_LOG_LEVEL', 'NOTSET')
log.setLevel(log_level)
