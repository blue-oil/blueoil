import tensorflow as tf

from lmnet.networks.classification.lmnet_v0 import LmnetV0Quantize
from lmnet.networks.classification.lmnet_v1 import LmnetV1Quantize
from lmnet.networks.classification.darknet import DarknetQuantize
from lmnet.networks.object_detection.lm_fyolo import LMFYoloQuantize
from lmnet.networks.object_detection.yolo_v2_quantize import YoloV2Quantize
from lmnet.networks.segmentation.lm_segnet_v0 import LmSegnetV0Quantize
from lmnet.networks.segmentation.lm_segnet_v1 import LmSegnetV1Quantize
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)


def test_required_arguments():
    model_classes = [
        LmnetV0Quantize,
        LmnetV1Quantize,
        DarknetQuantize,
        LMFYoloQuantize,
        YoloV2Quantize,
        LmSegnetV0Quantize,
        LmSegnetV1Quantize,
    ]

    network_kwargs = {
        'activation_quantizer': linear_mid_tread_half_quantizer,
        'batch_size': 10,
        'data_format': 'NHWC',
        'image_size': [128, 128],
        'optimizer_class': tf.train.GradientDescentOptimizer,
        'quantize_first_convolution': True,
        'weight_decay_rate': 0.0005,
        'weight_quantizer': binary_mean_scaling_quantizer,
    }

    for model in model_classes:
        quantizer = model(
            classes = ['accordion', 'airplanes', 'anchor'],
            is_debug = True,
            network_kwargs = **network_kwargs
        )

        assert quantizer.first_layer_name is not None
        assert quantizer.last_layer_name is not None


if __name__ == '__main__':
    test_required_arguments()
