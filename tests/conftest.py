import pytest

from lightwood.config.config import CONFIG

pytest_plugins = ("plugin",)


@pytest.fixture(autouse=True)
def config(tmp_path):
    CONFIG.ENABLE_DROPOUT = False
