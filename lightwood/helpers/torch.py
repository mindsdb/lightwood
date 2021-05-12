import functools
import itertools
import warnings
import torch
from torch.nn.functional import pad
from torch.utils.data import SequentialSampler
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


class NotNullSampler(SequentialSampler):
    def __init__(self, ds):
        super().__init__(self)
        self.incomplete_rows = self.get_incomplete_rows(ds)

    def get_incomplete_rows(self, ds):
        temporal_features = [feat for feat in ds.input_features if feat['type'] == 'time_series']
        incomplete_rows = set()
        for tf in temporal_features:
            incomplete_rows.add([idx for idx, row in ds.data_frame.iterrows() if not all(row['T'])])
        return incomplete_rows

    def __iter__(self):
        for idx in range(len(self.data_source)):
            if idx not in self.incomplete_rows:
                yield idx

class LightwoodAutocast:
    """
    Equivalent to torch.cuda.amp.autocast, but checks device compute capability
    to activate the feature only when the GPU has tensor cores to leverage AMP.
    """
    active = False
    def __init__(self, enabled=True):
        self.major = 0  # GPU major version
        torch_version = [int(i) for i in torch.__version__.split('.')[:-1]]

        if not enabled or not torch.cuda.is_available() or torch_version[0] < 1 or torch_version[1] < 6:
            self._enabled = False
        else:
            device, _ = get_devices()
            if device.type == 'cuda':
                # tensor cores only exist from 7 onwards
                # if this is not the case, then AMP is unnecessary overhead
                self.major, _ = torch.cuda.get_device_capability(device)
                self._enabled = enabled if self.major > 6 else False
            else:
                self._enabled = False  # gpu is available but cpu is forced

        self.prev = self._enabled  # necessary reference to exit
        LightwoodAutocast.active = self._enabled

    def __enter__(self):
        if self._enabled:
            self.prev = torch.is_autocast_enabled()
            torch.set_autocast_enabled(self._enabled)
            torch.autocast_increment_nesting()

    def __exit__(self, *args):
        if self._enabled:
            # Drop the cache when we exit to a nesting level that's outside any instance of autocast
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