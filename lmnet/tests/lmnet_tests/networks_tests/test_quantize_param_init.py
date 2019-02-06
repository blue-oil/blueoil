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

    for model in model_classes:
        quantizer = model(
            classes=['accordion', 'airplanes', 'anchor'],
            is_debug=True,
            activation_quantizer=linear_mid_tread_half_quantizer,
            batch_size=10,
            data_format='NHWC',
            image_size=[128, 128],
            optimizer_class=tf.train.GradientDescentOptimizer,
            weight_quantizer=binary_mean_scaling_quantizer,
            activation_quantizer_kwargs={
                'bit': 2,
                'max_value': 2
            }
        )

        assert quantizer.first_layer_name is not None
        assert quantizer.last_layer_name is not None


def test_quantized_layers():
    model_classes = [
        LmnetV0Quantize,
        LmnetV1Quantize,
        DarknetQuantize,
        LMFYoloQuantize,
        YoloV2Quantize,
        LmSegnetV0Quantize,
        LmSegnetV1Quantize,
    ]

    quantizer_name = "QTZ_binary_mean_scaling"

    for model in model_classes:
        quantizer = model(
            classes=['accordion', 'airplanes', 'anchor'],
            is_debug=True,
            activation_quantizer=linear_mid_tread_half_quantizer,
            batch_size=10,
            data_format='NHWC',
            image_size=[32, 32],
            optimizer_class=tf.train.GradientDescentOptimizer,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            weight_quantizer=binary_mean_scaling_quantizer,
            activation_quantizer_kwargs={
                'bit': 2,
                'max_value': 2
            }
        )

        base, graph = quantizer.base(tf.ones([10, 32, 32, 3]), True)
        op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
        scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
        assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)
        tf.reset_default_graph()

        with tf.variable_scope("notQuantizeFirstLayer"):
            quantizer = model(
                classes=['accordion', 'airplanes', 'anchor'],
                is_debug=True,
                activation_quantizer=linear_mid_tread_half_quantizer,
                batch_size=10,
                data_format='NHWC',
                image_size=[32, 32],
                optimizer_class=tf.train.GradientDescentOptimizer,
                quantize_first_convolution=False,
                quantize_last_convolution=True,
                weight_quantizer=binary_mean_scaling_quantizer,
                activation_quantizer_kwargs={
                    'bit': 2,
                    'max_value': 2
                }
            )

            base, graph = quantizer.base(tf.ones([10, 32, 32, 3]), True)
            op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
            assert not any(quantizer.first_layer_name in op and quantizer_name in op for op in op_name_list)

            op_name_list = [op_name for op_name in op_name_list if quantizer.first_layer_name not in op_name]
            scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
            assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)

        with tf.variable_scope("notQuantizeLastLayer"):
            quantizer = model(
                classes=['accordion', 'airplanes', 'anchor'],
                is_debug=True,
                activation_quantizer=linear_mid_tread_half_quantizer,
                batch_size=10,
                data_format='NHWC',
                image_size=[32, 32],
                optimizer_class=tf.train.GradientDescentOptimizer,
                quantize_first_convolution=True,
                quantize_last_convolution=False,
                weight_quantizer=binary_mean_scaling_quantizer,
                activation_quantizer_kwargs={
                    'bit': 2,
                    'max_value': 2
                }
            )

            base, graph = quantizer.base(tf.ones([10, 32, 32, 3]), True)
            op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
            assert not any(quantizer.last_layer_name in op and quantizer_name in op for op in op_name_list)

            op_name_list = [op_name for op_name in op_name_list if quantizer.last_layer_name not in op_name]
            scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
            assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


if __name__ == '__main__':
    test_required_arguments()
