import tensorflow as tf

from lmnet.networks.classification.lmnet_v0 import LmnetV0Quantize
from lmnet.networks.classification.lmnet_v1 import LmnetV1Quantize
from lmnet.networks.classification.darknet import DarknetQuantize
from lmnet.networks.object_detection.lm_fyolo import LMFYoloQuantize
from lmnet.networks.object_detection.yolo_v2_quantize import YoloV2Quantize
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)
from lmnet.networks.quantize_param_init import QuantizeParamInit


class TestBaseNetwork():
    def __init__(self, *args, **kwargs):
        self.first_layer_name = "conv1"
        self.last_layer_name = "conv2"

    def base(self, images, is_training):
        with tf.variable_scope(self.first_layer_name):
            tf.layers.dense(images)

        with tf.variable_scope(self.last_layer_name):
            tf.layers.dense(images)

        return tf.constant(0)


class TestNetwork(QuantizeParamInit, TestBaseNetwork):
    def base(self, images, is_training):
        with tf.variable_scope("", custom_getter=self.custom_getter):
            return super().base(images, is_training)


def test_required_arguments():
    model_classes = [
        LmnetV0Quantize,
        LmnetV1Quantize,
        DarknetQuantize,
        LMFYoloQuantize,
        YoloV2Quantize,
        TestNetwork,
    ]

    for model in model_classes:
        network = model(
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

        assert network.first_layer_name is not None
        assert network.last_layer_name is not None


def test_quantized_both_layers():
    model_classes = [
        LmnetV0Quantize,
        LmnetV1Quantize,
        DarknetQuantize,
        LMFYoloQuantize,
        YoloV2Quantize,
    ]

    quantizer_name = "QTZ_binary_mean_scaling"
    network_kwargs = {
        "classes": ['accordion', 'airplanes', 'anchor'],
        "is_debug": True,
        "activation_quantizer": linear_mid_tread_half_quantizer,
        "batch_size": 10,
        "data_format": 'NHWC',
        "image_size": [32, 32],
        "optimizer_class": tf.train.GradientDescentOptimizer,
        "quantize_first_convolution": True,
        "quantize_last_convolution": True,
        "weight_quantizer": binary_mean_scaling_quantizer,
        "activation_quantizer_kwargs": {
            'bit': 2,
            'max_value': 2
        }
    }

    for model in model_classes:
        network_kwargs["quantize_first_convolution"] = True
        network_kwargs["quantize_last_convolution"] = True

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            network1 = model(**network_kwargs)

            network1.base(tf.ones([10, 32, 32, 3]), True)
            graph1 = tf.get_default_graph()
            op_name_list = [op.name for op in graph1.get_operations() if "kernel" in op.name]
            scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
            assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)

        tf.reset_default_graph()
        network_kwargs["quantize_first_convolution"] = False

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            network2 = model(**network_kwargs)

            network2.base(tf.ones([10, 32, 32, 3]), True)
            graph2 = tf.get_default_graph()
            op_name_list = [op.name for op in graph2.get_operations() if "kernel" in op.name]
            assert not any(network2.first_layer_name in op and quantizer_name in op for op in op_name_list)

            op_name_list = [op_name for op_name in op_name_list if network2.first_layer_name not in op_name]
            scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
            assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)

        tf.reset_default_graph()
        network_kwargs["quantize_first_convolution"] = True
        network_kwargs["quantize_last_convolution"] = False

        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            network3 = model(**network_kwargs)

            network3.base(tf.ones([10, 32, 32, 3]), True)
            graph3 = tf.get_default_graph()
            op_name_list = [op.name for op in graph3.get_operations() if "kernel" in op.name]
            assert not any(network3.last_layer_name in op and quantizer_name in op for op in op_name_list)

            op_name_list = [op_name for op_name in op_name_list if network3.last_layer_name not in op_name]
            scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
            assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)

        tf.reset_default_graph()


if __name__ == '__main__':
    test_required_arguments()
