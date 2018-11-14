import tensorflow as tf

class BaseQuantize(object):
    

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            quantize_first_convolution=None,
            quantize_last_convolution=None,
    ):

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)

        self.quantize_first_convolution = quantize_first_convolution
        self.first_layer_name = None
        self.quantize_last_convolution = quantize_last_convolution                                            
