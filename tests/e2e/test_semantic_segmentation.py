import pytest

from conftest import run_all_steps


@pytest.mark.parametrize(
    "config_file", [
        "camvid_custom_semantic_segmentation.py",
        "camvid_custom_semantic_segmentation_has_validation.py",
    ]
)
def test_semantic_segmentation(init_env, config_file):
    """Run Blueoil test of semantic segmentation"""
    run_all_steps(init_env, config_file)
