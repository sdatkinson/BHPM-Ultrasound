# File: conftest.py
# Created Date: 2020-04-12
# Author: Steven Atkinson (steven@atkinson.mn)

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--rundemos", action="store_true", default=False, help="run demo scripts"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "demo: mark test as a demo script")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rundemos"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_demos = pytest.mark.skip(reason="need --rundemos option to run")
    for item in items:
        if "demo" in item.keywords:
            item.add_marker(skip_demos)
