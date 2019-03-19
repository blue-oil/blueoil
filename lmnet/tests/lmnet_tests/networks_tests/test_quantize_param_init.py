import pytest
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

pytestmark = pytest.mark.usefixtures("reset_default_graph", "set_test_environment")


def quantizer_name_func():
    return "QTZ_binary_mean_scaling"


def network_kwargs_func(first_quantize=True, last_quantize=True):
    network_kwargs = {
        "classes": ['accordion', 'airplanes', 'anchor'],
        "is_debug": False,
        "activation_quantizer": linear_mid_tread_half_quantizer,
        "batch_size": 10,
        "data_format": 'NHWC',
        "image_size": [32, 32],
        "optimizer_class": tf.train.GradientDescentOptimizer,
        "quantize_first_convolution": first_quantize,
        "quantize_last_convolution": last_quantize,
        "weight_quantizer": binary_mean_scaling_quantizer,
        "activation_quantizer_kwargs": {
            'bit': 2,
            'max_value': 2
        }
    }
    return network_kwargs


def test_lmnetv0quantize_quantize_both():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func()

    network = LmnetV0Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmnetv0quantize_quantize_first():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(last_quantize=False)

    network = LmnetV0Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.last_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.last_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmnetv0quantize_quantize_last():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(first_quantize=False)

    network = LmnetV0Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.first_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.first_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmnetv1quantize_quantize_both():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func()

    network = LmnetV1Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmnetv1quantize_quantize_first():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(last_quantize=False)

    network = LmnetV1Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.last_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.last_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmnetv1quantize_quantize_last():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(first_quantize=False)

    network = LmnetV1Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.first_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.first_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_darknetquantize_quantize_both():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func()

    network = DarknetQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_darknetquantize_quantize_first():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(last_quantize=False)

    network = DarknetQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.last_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.last_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_darknetquantize_quantize_last():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(first_quantize=False)

    network = DarknetQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.first_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.first_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmfyoloquantize_quantize_both():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func()

    network = LMFYoloQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmfyoloquantize_quantize_first():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(last_quantize=False)

    network = LMFYoloQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.last_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.last_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_lmfyoloquantize_quantize_last():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(first_quantize=False)

    network = LMFYoloQuantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.first_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.first_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_yolov2quantize_quantize_both():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func()

    network = YoloV2Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    op_name_list = [op.name for op in graph.get_operations() if "kernel" in op.name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_yolov2quantize_quantize_first():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(last_quantize=False)

    network = YoloV2Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.last_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.last_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


def test_yolov2quantize_quantize_last():
    quantizer_name = quantizer_name_func()
    network_kwargs = network_kwargs_func(first_quantize=False)

    network = YoloV2Quantize(**network_kwargs)
    network.base(tf.ones([10, 32, 32, 3]), True)
    graph = tf.get_default_graph()

    ops = graph.get_operations()
    op_name_list = [op.name for op in ops if "kernel" in op.name]
    assert not any(network.first_layer_name in op and quantizer_name in op for op in op_name_list)

    op_name_list = [op_name for op_name in op_name_list if network.first_layer_name not in op_name]
    scope_name_list = list(set([op.split("/")[0] for op in op_name_list]))
    assert all(any(scope in op and quantizer_name in op for op in op_name_list) for scope in scope_name_list)


if __name__ == '__main__':
    test_lmnetv0quantize_quantize_both()
    test_lmnetv0quantize_quantize_first()
    test_lmnetv0quantize_quantize_last()
    test_lmnetv1quantize_quantize_both()
    test_lmnetv1quantize_quantize_first()
    test_lmnetv1quantize_quantize_last()
    test_darknetquantize_quantize_both()
    test_darknetquantize_quantize_first()
    test_darknetquantize_quantize_last()
    test_lmfyoloquantize_quantize_both()
    test_lmfyoloquantize_quantize_first()
    test_lmfyoloquantize_quantize_last()
    test_yolov2quantize_quantize_both()
    test_yolov2quantize_quantize_first()
    test_yolov2quantize_quantize_last()
