import tempfile

import pytest


@pytest.fixture
def temp_dir():
    temp = tempfile.TemporaryDirectory()
    yield temp.name
    temp.cleanup()
