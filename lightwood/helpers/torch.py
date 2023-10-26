import functools
import torch
from torch.nn.functional import pad
from lightwood.helpers.device import get_devices


def concat_vectors_and_pad(vec_list, max_):
    """
    Concatenates a list of input vectors and pads them to match a specified maximum
    length.

    This function takes a list of input vectors, concatenates them along a specified
    dimension (dim=0), and then pads the concatenated vector to achieve a specified
    maximum length. The padding is done with zeros.

    Args:
        vec_list (list of torch.Tensor): List of input vectors to concatenate and pad.
        max_ (int): The maximum length of the concatenated and padded vector.

    Returns:
        torch.Tensor: The concatenated and padded vector.

    Raises:
        AssertionError: If the length of 'vec_list' is not greater than 0, or if it
                        exceeds 'max_len', or if 'max_len' is not greater than 0.

    Example:
        >>> input_tensors = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
        >>> max_length = 5
        >>> concatenated_padded = concat_vectors_and_pad(input_tensors, max_length)
        >>> print(concatenated_padded)
        tensor([1, 2, 3, 4, 5])
    """
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

    **Attributes:**

    * `active` (bool): Whether AMP is currently active. This attribute is at the class
    level

    **Usage:**

    ```python
    >>> import lightwood.helpers.torch as lt
    >>> with lt.LightwoodAutocast():
    ...     # This code will be executed in AMP mode.
    ...     pass
    """
    active = False

    def __init__(self, enabled=True):
        """
        Initializes the context manager for Automatic Mixed Precision (AMP) functionality.

        Args:
            enabled (bool, optional): Whether to enable AMP. Defaults to True.
        """
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
        """
        * `__enter__()`: Enters the context manager and enables AMP if it is not already enabled.
        """
        if self._enabled:
            self.prev = torch.is_autocast_enabled()
            torch.set_autocast_enabled(self._enabled)
            torch.autocast_increment_nesting()

    def __exit__(self, *args):
        """
        * `__exit__()`: Exits the context manager and disables AMP.
        """
        if self._enabled:
            # Drop the cache when we exit to a nesting level that's outside any instance of autocast
            if torch.autocast_decrement_nesting() == 0:
                torch.clear_autocast_cache()
            torch.set_autocast_enabled(self.prev)
        return False

    def __call__(self, func):
        """
        * `__call__(self, func)`: Returns a decorated function that enables AMP when it is called.
        """
        @functools.wraps(func)
        def decorate_autocast(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_autocast
