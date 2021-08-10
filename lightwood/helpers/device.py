import torch
import os
from random import randint
from torch.cuda import device_count, get_device_capability


def is_cuda_compatible():
    compatible_device_count = 0
    if torch.version.cuda is not None:
        for d in range(device_count()):
            capability = get_device_capability(d)
            major = capability[0]
            minor = capability[1]
            current_arch = major * 10 + minor
            min_arch = min((int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()), default=35)
            if (not current_arch < min_arch
                    and not torch._C._cuda_getCompiledVersion() <= 9000):
                compatible_device_count += 1

    if compatible_device_count > 0:
        return True
    return False


def get_devices():
    if torch.cuda.is_available() and is_cuda_compatible():
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
