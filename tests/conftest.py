"""
Shared pytest fixtures.

Every test in this suite takes a single argument, `tmp`: a fresh directory it
may fill with images, bundles and pickles. The files also carry their own
`main()` runner so they work without pytest at all (`python tests/test_bundle.py`),
and that runner hands each test a `tempfile.mkdtemp()`. This fixture is the
pytest half of the same contract -- without it, `python -m pytest tests` errors
out on every test with "fixture 'tmp' not found" even though the suite is fine.

`tmp` is a plain str, not a Path: the tests join it with os.path.join and pass it
to code that takes string paths, and pytest's own tmp_path is a Path.
"""

import os
import sys

import pytest

# The tests import `arcana.*`, and are run from wherever the developer happens to
# be standing. Each file already does this for its standalone runner; doing it
# here too means collection works even if a file is imported before its own
# sys.path line runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tmp(tmp_path):
    """A fresh scratch directory, as a string path."""
    return str(tmp_path)
