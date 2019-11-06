import pytest

from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "caltech101_classification.yml",
        "caltech101_classification_has_validation.yml",
        "delta_mark_classification.yml",
        "delta_mark_classification_has_validation.yml",
    ]
)
def test_classification(init_env, config_file):
    """Run Blueoil test of classification"""
    run_all_steps(init_env, config_file)
