from lightwood.config.config import CONFIG
from psutil import virtual_memory
import torch


def calculate_batch_size():
    device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
    if CONFIG.USE_DEVICE is not None:
        device_str = CONFIG.USE_DEVICE

def get_cpu_available_memory():
    mem = virtual_memory()
    return mem.available, mem.total

def get_gpu_available_memory():
    total = torch.cuda.get_device_properties(0).total_memory
    cached = torch.cuda.memory_cached(0)
    allocated = torch.cuda.memory_allocated(0)
    return allocated, total

def estimate_size(net, sample, device):
        if 'cuda' in str(device):
            get_available_memory = get_gpu_available_memory
        else:
            get_available_memory = get_cpu_available_memory

        before_memory, total = get_available_memory()

        net = net.train()
        net(sample)

        after_memory, total = get_available_memory()
        net.zero_grad()

        network_max_memory = after_memory - before_memory
        percentage_left = (total - after_memory) / total

        return network_max_memory, percentage_left
