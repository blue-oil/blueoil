"""
Auto config generator (for unit test)

usage:
  python make_test_config.py
"""


import shutil

from blueoil.generate_lmnet_config import generate


output_files = [
    'caltech101_classification',
    'caltech101_classification_has_validation',
    'delta_mark_classification',
    'delta_mark_classification_has_validation',
    'delta_mark_object_detection',
    'delta_mark_object_detection_has_validation',
    'openimagesv4_object_detection',
    'openimagesv4_object_detection_has_validation',
    'camvid_custom_semantic_segmentation',
    'camvid_custom_semantic_segmentation_has_validation',
    'mscoco2017_single_person_pose_estimation',
    'mscoco2017_single_person_pose_estimation_has_validation'
]


task_types = [
    'classification',
    'classification',
    'classification',
    'classification',
    'object_detection',
    'object_detection',
    'object_detection',
    'object_detection',
    'semantic_segmentation',
    'semantic_segmentation',
    'keypoint_detection',
    'keypoint_detection',
]


network_names = [
    'LmnetV1Quantize',
    'LmnetV1Quantize',
    'LmnetV1Quantize',
    'LmnetV1Quantize',
    'LMFYoloQuantize',
    'LMFYoloQuantize',
    'LMFYoloQuantize',
    'LMFYoloQuantize',
    'LmSegnetV1Quantize',
    'LmSegnetV1Quantize',
    'LmSinglePoseV1Quantize',
    'LmSinglePoseV1Quantize',
]


dataset_formats = [
    'Caltech101',
    'Caltech101',
    'DeLTA-Mark for Classification',
    'DeLTA-Mark for Classification',
    'DeLTA-Mark for Object Detection',
    'DeLTA-Mark for Object Detection',
    'OpenImagesV4',
    'OpenImagesV4',
    'CamvidCustom',
    'CamvidCustom',
    'Mscoco for Single-Person Pose Estimation',
    'Mscoco for Single-Person Pose Estimation',
]


dataset_train_paths = [
    'dummy_classification',
    'dummy_classification',
    'custom_delta_mark_classification/for_train',
    'custom_delta_mark_classification/for_train',
    'custom_delta_mark_object_detection/for_train',
    'custom_delta_mark_object_detection/for_train',
    'custom_open_images_v4_bounding_boxes/for_train',
    'custom_open_images_v4_bounding_boxes/for_train',
    'camvid_custom',
    'camvid_custom',
    'MSCOCO_2017',
    'MSCOCO_2017',
]


dataset_test_paths = [
    None,
    'dummy_classification',
    None,
    'custom_delta_mark_classification/for_validation',
    None,
    'custom_delta_mark_object_detection/for_validation',
    None,
    'custom_open_images_v4_bounding_boxes/for_validation',
    None,
    'camvid_custom',
    None,
    'MSCOCO_2017',
]


trainer_batch_sizes = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


trainer_epochs = [
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]


trainer_optimizers = [
    'Adam',
    'Adam',
    'Adam',
    'Adam',
    'Momentum',
    'Momentum',
    'Momentum',
    'Momentum',
    'Adam',
    'Adam',
    'Adam',
    'Adam',
]


trainer_lr_schedules = [
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
    'constant',
]


trainer_initial_lrs = [
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
    0.001,
]


network_quantize_first_convolution = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
]


common_image_sizes = [
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [128, 128],
    [160, 160],
    [160, 160],
]


common_is_pretrain_model = [
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]


common_enable_prefetch = [
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
    True,
]

common_data_augmentation = {'Blur': [('value', (0, 1))]}


def learning_settings_to_config_py(index):
    model_name = output_files[index]

    config = {
        'model_name': model_name,
        'task_type': task_types[index],
        'network_name': network_names[index],
        'dataset': {
            'format': dataset_formats[index],
            'train_path': dataset_train_paths[index],
            'test_path': dataset_test_paths[index],
        },
        'trainer': {
            'batch_size': trainer_batch_sizes[index],
            'epochs': trainer_epochs[index],
            'optimizer': trainer_optimizers[index],
            'learning_rate_schedule': trainer_lr_schedules[index],
            'initial_learning_rate': trainer_initial_lrs[index],
            'save_checkpoint_steps': 1000,
            'keep_checkpoint_max': 5,
        },
        'network': {
            'quantize_first_convolution': network_quantize_first_convolution[index],
        },
        'common': {
            'image_size': common_image_sizes[index],
            'pretrain_model': common_is_pretrain_model[index],
            'dataset_prefetch': common_enable_prefetch[index],
            'data_augmentation': common_data_augmentation,
        },
    }

    tmpfile = generate(config)
    shutil.copy(tmpfile, model_name + ".py")


def main():
    for index in range(0, len(output_files)):
        learning_settings_to_config_py(index)


if __name__ == '__main__':
    main()
