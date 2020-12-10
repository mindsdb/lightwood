import functools
import warnings
import torch
from torch.nn.functional import pad
from lightwood.helpers.device import get_devices


def concat_vectors_and_pad(vec_list, max_):
    assert len(vec_list) > 0
    assert len(vec_list) <= max_
    assert max_ > 0

    cat_vec = torch.cat(list(vec_list), dim=0)

    pad_size = max_ - len(vec_list)
    padding = (0, pad_size * vec_list[0].size(0))
    padded = pad(cat_vec[None], padding, 'constant', 0)[0]

    return padded


def average_vectors(vec_list):
    assert len(vec_list) > 0
    return torch.cat([emb[None] for emb in vec_list], dim=0).mean(0)


class LightwoodAutocast:
    """
    Equivalent to torch.cuda.amp.autocast, but checks device compute capability
    to activate the feature only when the GPU has tensor cores to leverage AMP.
    """
    def __init__(self, enabled=True):
        self.major = 0  # GPU major version

        if enabled and not torch.cuda.is_available():
            self._enabled = False
        else:
            device, _ = get_devices()
            if device.type == 'cuda':
                self.major, _ = torch.cuda.get_device_capability(device)
                self._enabled = enabled
            else:
                self._enabled = False  # gpu is available but cpu is forced

        self.prev = self._enabled

    def __enter__(self):
        # tensor cores only exist from 7 onwards
        # if this is not the case, then AMP is unnecessary overhead
        if self.major > 6:
            self.prev = torch.is_autocast_enabled()
            torch.set_autocast_enabled(self._enabled)
            torch.autocast_increment_nesting()

    def __exit__(self, *args):
        if self.major > 6:
            # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_enabled(self.prev)
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_autocast