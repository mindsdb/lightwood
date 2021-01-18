import torch


class CONFIG:
    """General"""
    USE_CUDA = False
    if torch.cuda.device_count() > 0:
        try:
            torch.ones(1).cuda()
            # pytorch wheels support compute capability >= 3.7
            major, minor = torch.cuda.get_device_capability()
            if major + minor/10 >= 3.7:
                USE_CUDA = True
            else:
                USE_CUDA = False
        except Exception as e:
            USE_CUDA = False
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
