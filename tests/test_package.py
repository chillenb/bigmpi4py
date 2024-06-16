from __future__ import annotations

import importlib.metadata

import mpix4py as m


def test_version():
    assert importlib.metadata.version("mpix4py") == m.__version__
