import torch
import os
from random import randint


def get_devices():
    if torch.cuda.is_available() and float(torch.version.cuda) >= 9:
        device_str = "cuda"
        available_devices = torch.cuda.device_count()

        if available_devices > 1:
            if os.environ.get('RANDOM_GPU', False) in ['1', 'true', 'True', True, 1]:
                device_str = 'cuda:' + str(randint(0, available_devices - 1))
                available_devices = 1
    else:
        device_str = "cpu"
        available_devices = 0

    return torch.device(device_str), available_devices
