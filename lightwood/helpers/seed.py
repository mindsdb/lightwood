import random
import torch
import numpy as np
import mxnet as mx


def seed(seed_nr: int) -> None:
    torch.manual_seed(seed_nr)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_nr)
    random.seed(seed_nr)
    mx.random.seed(seed_nr)
