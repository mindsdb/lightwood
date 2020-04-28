import torch

from lightwood.config.config import CONFIG


def get_devices():
    device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
    if CONFIG.USE_DEVICE is not None:
        device_str = CONFIG.USE_DEVICE
    device = torch.device(device_str)

    available_devices = 1
    if device_str == 'cuda':
        available_devices = torch.cuda.device_count()

    return device, available_devices
