import torch


class CONFIG:
    """General"""
    USE_CUDA = False
    if torch.cuda.device_count() > 0:
        USE_CUDA = True
    USE_DEVICE = None

    # Enable deterministic cuda flag and use seeds everywhere (static or based on features of the dataset)
    HELPER_MIXERS = True



    # Here for development purposes
    USE_PROBABILISTIC_LINEAR = False
    ENABLE_DROPOUT = False
    FORCE_HELPER_MIXERS = False

    MONITORING = {
        'epoch_loss': False
        ,'batch_loss': False
        ,'network_heatmap': False
    }
