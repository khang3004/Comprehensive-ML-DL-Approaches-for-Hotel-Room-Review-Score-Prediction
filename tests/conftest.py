"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def models_dir():
    """Return the models directory."""
    return Path(__file__).parent.parent / "models"


@pytest.fixture(scope="session")
def results_dir():
    """Return the results directory."""
    return Path(__file__).parent.parent / "results"
