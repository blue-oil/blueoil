import pytest

from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "openimagesv4_object_detection.yml",
        "openimagesv4_object_detection_has_validation.yml",
        "delta_mark_object_detection.yml",
        "delta_mark_object_detection_has_validation.yml",
    ]
)
def test_object_detection(init_env, config_file):
    """Run Blueoil test of object detection"""
    run_all_steps(init_env, config_file)
