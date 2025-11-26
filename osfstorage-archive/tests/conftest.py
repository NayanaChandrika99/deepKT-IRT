"""Test fixtures"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    """Path for raw data dir"""
    return Path.home() / "data-store" / "edm-cup-2023"


@pytest.fixture(scope="session")
def processed_data_dir() -> Path:
    """Path for processed data dir"""
    return Path.home() / "data-store" / "edm-cup-2023" / "processed"
