from __future__ import annotations

import importlib.metadata

import bigmpi4py as m


def test_version():
    assert importlib.metadata.version("bigmpi4py") == m.__version__
