import requests
from lightwood.helpers.log import log
import os


def read_from_path_or_url(path: str, load_from_path):
    if path.startswith('http'):
        response = requests.get(path)
        with open(path.split('/')[-1], 'wb') as f:
            f.write(response.content)
        try:
            return load_from_path(path.split('/')[-1])
        except Exception as e:
            log.error(e)
        finally:
            os.remove(path.split('/')[-1])
    else:
        # Will automatically resample to 22.05kHz and convert to mono
        return load_from_path(path)

