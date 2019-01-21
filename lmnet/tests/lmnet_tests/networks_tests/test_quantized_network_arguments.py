import tensorflow as tf

from lmnet.networks.classification import (
    LmnetV1Quantize,
    LmnetV0Quantize,
    DarknetQuantize,
)
from lmnet.networks.object_detection import (
    LMFYoloQuantize,
    YoloV2Quantize,
)
from lmnet.networks.segmentation import (
    LmSegnetV0Quantize,
    LmSegnetV1Quantize,
)
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)


def test_required_arguments():
    model_classes = [
        LmnetV1Quantize,
        LmnetV0Quantize,
        DarknetQuantize,
        LMFYoloQuantize,
        YoloV2Quantize,
        LmSegnetV0Quantize,
        LmSegnetV1Quantize,
    ]

    network_kwargs = {
        'ACTIVATION_QUANTIZER': linear_mid_tread_half_quantizer,
        'ACTIVATION_QUANTIZER_KWARGS': {'bit': 2, 'max_value': 2},
        'BATCH_SIZE': 10,
        'DATA_FORMAT': 'NHWC',
        'IMAGE_SIZE': [128, 128],
        'OPTIMIZER_CLASS': tf.train.AdamOptimizer,
        'OPTIMIZER_KWARGS': {'momentum': 0.9},
        'QUANTIZE_FIRST_CONVOLUTION': False,
        'WEIGHT_DECAY_RATE': 0.0005,
        'WEIGHT_QUANTIZER': binary_mean_scaling_quantizer,
        'WEIGHT_QUANTIZER_KWARGS': {},
    }

    for model in model_classes:
        quantizer = model(
            ['accordion', 'airplanes', 'anchor'],
            True,
            network_kwargs
        )

        assert quantizer.first_layer_name is not None
        assert quantizer.last_layer_name is not None


if __name__ == '__main__':
    test_required_arguments()
