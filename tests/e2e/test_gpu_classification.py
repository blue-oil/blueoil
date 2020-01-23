import pytest

from conftest import is_gpu_available
from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "caltech101_classification.yml",
    ]
)
def test_gpu_classification(init_env, config_file):
    """Run Blueoil test of classification"""
    is_gpu_available()
    run_all_steps(init_env, config_file)
