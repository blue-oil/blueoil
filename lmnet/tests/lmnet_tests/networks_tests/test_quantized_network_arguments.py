import tensorflow as tf
from lmnet.networks.classification.lmnet_quantize import LmnetQuantize
from lmnet.quantizations import (
    binary_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

def test_required_arguments():    
    quantizer = LmnetQuantize(classes=['accordion',
                                       'airplanes',
                                       'anchor'],
                              is_debug=True,
                              { 'ACTIVATION_QUANTIZER': linear_mid_tread_half_quantizer,
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
    )

    assert quantizer.first_layer_name is not None
    assert quantizer.last_layer_name is not None


if __name__ == '__main__':
    test_required_arguments()
