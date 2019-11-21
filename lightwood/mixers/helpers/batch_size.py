from lightwood.config.config import CONFIG


def calculate_batch_size():
    device_str = "cuda" if CONFIG.USE_CUDA else "cpu"
    if CONFIG.USE_DEVICE is not None:
        device_str = CONFIG.USE_DEVICE
