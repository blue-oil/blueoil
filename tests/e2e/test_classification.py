import pytest

from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "caltech101_classification.py",
        "caltech101_classification_has_validation.py",
        "delta_mark_classification.py",
        "delta_mark_classification_has_validation.py",
    ]
)
def test_classification(init_env, config_file):
    """Run Blueoil test of classification"""
    run_all_steps(init_env, config_file)
