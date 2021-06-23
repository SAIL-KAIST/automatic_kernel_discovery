import logging
from os import environ

logging.basicConfig(level=environ.get('LOG_LEVEL', 'INFO'))
logging.getLogger('tensorflow').setLevel(logging.ERROR)