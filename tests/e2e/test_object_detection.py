import pytest

from blueoil.generate_lmnet_config import generate
from conftest import run_all_steps


delta_mark_object_detection = {
    'model_name': 'delta_mark_object_detection',
    'task_type': 'object_detection',
    'network_name': 'LMFYoloQuantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'DeLTA-Mark for Object Detection',
        'train_path': 'custom_delta_mark_object_detection/for_train',
        'test_path': None
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Momentum',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [128, 128],
        'pretrain_model': False
    }
}


delta_mark_object_detection_has_validation = {
    'model_name': 'delta_mark_object_detection_has_validation',
    'task_type': 'object_detection',
    'network_name': 'LMFYoloQuantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'DeLTA-Mark for Object Detection',
        'train_path': 'custom_delta_mark_object_detection/for_train',
        'test_path': 'custom_delta_mark_object_detection/for_validation'
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Momentum',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [128, 128],
        'pretrain_model': False
    }
}


openimagesv4_object_detection = {
    'model_name': 'openimagesv4_object_detection',
    'task_type': 'object_detection',
    'network_name': 'LMFYoloQuantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'OpenImagesV4',
        'train_path': 'custom_open_images_v4_bounding_boxes/for_train',
        'test_path': None
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Momentum',
        'save_checkpoint_steps': 1000
    },
    'common': {
        'data_augmentation': {'Blur': [('value', (0, 1))]},
        'dataset_prefetch': True,
        'image_size': [128, 128],
        'pretrain_model': False
    }
}


openimagesv4_object_detection_has_validation = {
    'model_name': 'openimagesv4_object_detection_has_validation',
    'task_type': 'object_detection',
    'network_name': 'LMFYoloQuantize',
    'network': {
        'quantize_first_convolution': True
    },
    'dataset': {
        'format': 'OpenImagesV4',
        'train_path': 'custom_open_images_v4_bounding_boxes/for_train',
        'test_path': 'custom_open_images_v4_bounding_boxes/for_validation'
    },
    'trainer': {
        'batch_size': 1,
        'epochs': 1,
        'initial_learning_rate': 0.001,
        'keep_checkpoint_max': 5,
        'learning_rate_schedule': 'constant',
        'optimizer': 'Momentum',
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
        openimagesv4_object_detection,
        openimagesv4_object_detection_has_validation,
        delta_mark_object_detection,
        delta_mark_object_detection_has_validation,
    ]
)
def test_object_detection(init_env, config):
    """Run Blueoil test of object detection"""
    config_file = generate(config)
    run_all_steps(init_env, config_file)
