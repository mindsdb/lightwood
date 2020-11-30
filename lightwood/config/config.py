import torch


class CONFIG:
    """General"""
    USE_CUDA = False
    if torch.cuda.device_count() > 0:
        USE_CUDA = True
    USE_DEVICE = None

    # Development flags (maybe move to somewhere else later)
    USE_PROBABILISTIC_LINEAR = False
    ENABLE_DROPOUT = False
    FORCE_HELPER_MIXERS = False
    HELPER_MIXERS = True

    MONITORING = {
        'epoch_loss': False
        ,'batch_loss': False
        ,'network_heatmap': False
    }
