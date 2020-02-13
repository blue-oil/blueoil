import pytest

from blueoil.generate_lmnet_config import generate
from conftest import run_all_steps


caltech101_classification = {
    'model_name': 'caltech101_classification',
    'task_type': 'classification',
    'network_name': 'LmnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'Caltech101',
        'train_path': 'dummy_classification',
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


caltech101_classification_has_validation = {
    'model_name': 'caltech101_classification_has_validation',
    'task_type': 'classification',
    'network_name': 'LmnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'Caltech101',
        'train_path': 'dummy_classification',
        'test_path': 'dummy_classification'
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


delta_mark_classification = {
    'model_name': 'delta_mark_classification',
    'task_type': 'classification',
    'network_name': 'LmnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'DeLTA-Mark for Classification',
        'train_path': 'custom_delta_mark_classification/for_train',
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


delta_mark_classification_has_validation = {
    'model_name': 'delta_mark_classification_has_validation',
    'task_type': 'classification',
    'network_name': 'LmnetV1Quantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'DeLTA-Mark for Classification',
        'train_path': 'custom_delta_mark_classification/for_train',
        'test_path': 'custom_delta_mark_classification/for_validation'
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
        caltech101_classification,
        caltech101_classification_has_validation,
        delta_mark_classification,
        delta_mark_classification_has_validation,
    ]
)
def test_classification(init_env, config):
    """Run Blueoil test of classification"""
    config_file = generate(config)
    run_all_steps(init_env, config_file)
