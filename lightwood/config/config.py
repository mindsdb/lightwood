import torch


class CONFIG:
    """General"""
    USE_CUDA = False
    if torch.cuda.device_count() > 0:
        USE_CUDA = True

    USE_DEVICE = None
    CACHE_ENCODED_DATA = True

    # Enable deterministic cuda flag and use seeds everywhere (static or based on features of the dataset)
    DETERMINISTIC = True
    SELFAWARE = True
    HELPER_MIXERS = True
    FORCE_HELPER_MIXERS = False
    ENABLE_DROPOUT = True

    """Probabilistic FC layers"""
    USE_PROBABILISTIC_LINEAR = False # change weights in mixer to be probabilistic

    """Encoder options"""
    TRAIN_TO_PREDICT_TARGET = True
    MAX_ENCODER_TRAINING_TIME = 3600 * 2

    MONITORING = {
        'epoch_loss': False
        ,'batch_loss': False
        ,'network_heatmap': False
    }
