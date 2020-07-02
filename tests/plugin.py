"""This file allows to override pytest behavior,
add new command-line options and test markers."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests",
    )

    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run integration tests",
    )


def pytest_configure(config):
    if config.getoption("randomly_seed") == 'default':
        config.option.randomly_seed = 42

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration, only runs with --run-integration provided")


def pytest_collection_modifyitems(config, items):
    skip_marks = ['slow', 'integration']

    for mark in skip_marks:
        mark_option = f'--run-{mark}'
        if config.getoption(mark_option):
            continue

        skip = pytest.mark.skip(reason=f"need --run-{mark} option to run")
        for item in items:
            if mark in item.keywords:
                item.add_marker(skip)
