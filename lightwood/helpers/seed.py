import torch
import numpy as np
import random


def seed():
	torch.manual_seed(420)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(420)
	random.seed(420)
