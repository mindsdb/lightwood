import torch


class CONFIG:
    """General"""
    USE_CUDA = False
    if torch.cuda.device_count() > 0:
        USE_CUDA = True

    USE_DEVICE = None

    # Enable deterministic cuda flag and use seeds everywhere (static or based on features of the dataset)
    DETERMINISTIC = True
    SELFAWARE = True
    HELPER_MIXERS = True
    FORCE_HELPER_MIXERS = False
    ENABLE_DROPOUT = False

    """Probabilistic FC layers"""
    USE_PROBABILISTIC_LINEAR = False # change weights in mixer to be probabilistic

    MONITORING = {
        'epoch_loss': False
        ,'batch_loss': False
        ,'network_heatmap': False
    }

    QUANTILES = [0.05,0.95]
