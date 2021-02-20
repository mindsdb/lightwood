import os
import torch

from lightwood.config.config import CONFIG


def get_devices():
    if CONFIG.USE_CUDA and torch.cuda.is_available():
        device_str = "cuda"
        available_devices = torch.cuda.device_count()

        if available_devices > 1:
            round_robin = os.environ.get('ROUND_ROBIN_GPU', False)
            if round_robin in ['1', 'true', 'True', True]:
                round_robin = True
            else:
                round_robin = False

            if round_robin:
                current_gpu = int(os.environ.get('ROUND_ROBIN_GPU_CURRENT', '0'))
                device_str = 'cuda:' + str(current_gpu % available_devices)
                os.envrion['ROUND_ROBIN_GPU_CURRENT'] = (str(current_gpu+1))
    else:
        device_str = "cpu"
        available_devices = 1

    if CONFIG.USE_DEVICE is not None:
        device_str = CONFIG.USE_DEVICE
        if device_str != 'cuda':
            available_devices = 1

    device = torch.device(device_str)

    return device, available_devices
