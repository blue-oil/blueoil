import pytest

from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "mscoco2017_single_person_pose_estimation.yml",
        "mscoco2017_single_person_pose_estimation_has_validation.yml",
    ]
)
def test_keypoint_detection(init_env, config_file):
    """Run Blueoil test of keypoint detection"""
    run_all_steps(init_env, config_file)
