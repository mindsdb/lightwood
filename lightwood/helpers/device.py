import os
import torch
from random import randint

from lightwood.config.config import CONFIG


def get_devices():
    # Note: this initializes cuda (i.e. eats 2GB + of RAM)
    if CONFIG.USE_CUDA and torch.cuda.is_available():
        try:
            torch.ones(1).cuda().__repr__()
            cuda_certainly_works = True
        except:
            cuda_certainly_works = False
    else:
        cuda_certainly_works = False

    if CONFIG.USE_CUDA and torch.cuda.is_available() and cuda_certainly_works:
        device_str = "cuda"
        available_devices = torch.cuda.device_count()

        if available_devices > 1:
            if os.environ.get('RANDOM_GPU', False) in ['1', 'true', 'True', True, 1]:
                device_str = 'cuda:' + str(randint(0,available_devices-1))
                available_devices = 1
    else:
        device_str = "cpu"
        available_devices = 1

    if CONFIG.USE_DEVICE is not None and os.environ.get('RANDOM_GPU', False) not in ['1', 'true', 'True', True, 1]:
        device_str = CONFIG.USE_DEVICE
        if device_str != 'cuda':
            available_devices = 1

    device = torch.device(device_str)

    return device, available_devices
