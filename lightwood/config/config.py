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
    OVERSAMPLE = True
    SELFAWARE = False

    """Probabilistic FC layers"""
    USE_PROBABILISTIC_LINEAR = False # change weights in mixer to be probabilistic


    """Bayesian Network"""
    NUMBER_OF_PROBABILISTIC_MODELS = 2

    """Encoder options"""
    TRAIN_TO_PREDICT_TARGET = False
    MAX_ENCODER_TRAINING_TIME = 4 * 1800
