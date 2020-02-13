import pytest

from blueoil.generate_lmnet_config import generate
from conftest import run_all_steps


camvid_custom_semantic_segmentation = {
    'model_name': 'camvid_custom_semantic_segmentation',
    'task_type': 'semantic_segmentation',
    'network_name': 'LmSegnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'CamvidCustom',
        'train_path': 'camvid_custom',
        'test_path': None
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Adam',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [128, 128],
        'pretrain_model': False
    }
}


camvid_custom_semantic_segmentation_has_validation = {
    'model_name': 'camvid_custom_semantic_segmentation_has_validation',
    'task_type': 'semantic_segmentation',
    'network_name': 'LmSegnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'CamvidCustom',
        'train_path': 'camvid_custom',
        'test_path': 'camvid_custom'
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Adam',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [128, 128],
        'pretrain_model': False
    }
}


@pytest.mark.parametrize(
    "config", [
        camvid_custom_semantic_segmentation,
        camvid_custom_semantic_segmentation_has_validation,
    ]
)
def test_semantic_segmentation(init_env, config):
    """Run Blueoil test of semantic segmentation"""
    config_file = generate(config)
    run_all_steps(init_env, config_file)
