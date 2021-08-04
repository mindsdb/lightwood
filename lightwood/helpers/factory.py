from functools import partial
from typing import Callable, TypeVar, Type

# Factories are syntactic sugar but can introduce bugs, we should *only* use them for the public APIs


class Factory:
    self.generate_class = None

    def __init__(self, **kwargs):
        pass

    def generate(self) -> self.generate_class:
        return self.generate_class(**kwargs)


def gen_factory_func(func: Callable, **kwargs) -> Callable:
    return partial(func, **kwargs)
