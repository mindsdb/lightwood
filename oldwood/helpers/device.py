import os
import torch
from random import randint

from lightwood.config.config import CONFIG


def get_devices():
    if CONFIG.USE_CUDA and torch.cuda.is_available():
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
