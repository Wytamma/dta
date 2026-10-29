"""Shared pytest configuration: expose workflow/scripts as importable modules."""
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "workflow" / "scripts"
TEMPLATES_DIR = REPO_ROOT / "workflow" / "templates"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs BEAST and runs a short MCMC chain")


@pytest.fixture(scope="session")
def beast_bin():
    """Path to a BEAST executable, or skip the test."""
    import os

    candidate = os.environ.get("BEAST_BIN") or shutil.which("beast")
    if not candidate or not Path(candidate).exists():
        pytest.skip("BEAST not available (set BEAST_BIN or put beast on PATH)")
    return str(candidate)


@pytest.fixture(scope="session")
def template_path():
    return TEMPLATES_DIR / "beast_template.xml.j2"
