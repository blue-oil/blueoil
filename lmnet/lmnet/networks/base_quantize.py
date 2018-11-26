import tensorflow as tf


class BaseQuantize(object):
    """BaseQuantize.

    This base quantize class is a base class for quantized network of all kinds of task
    (classification, object detection, and segmentation).
    Every sub task's base network class with quantized layer should extend this class.

    Args:
        activation_quantizer (callable): An activation quantization function
        activation_quantizer_kwargs (dict): A dictionary of arguments for the activation quantization function
        weight_quantizer (callable): A weight quantization function
        weight_quantizer_kwargs (dict): A dictionary of arguments for the weight quantization function
        quantize_first_convolution (boolean): True if using quantization at the first layer, False otherwise
        quantize_last_convolution (boolean): True if using quantization at the last layer, False otherwise
    """

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

    @staticmethod
    def _quantized_variable_getter(getter,
                                   name,
                                   weight_quantization=None,
                                   quantize_first_convolution=None,
                                   quantize_last_convolution=None,
                                   first_layer_name=None,
                                   last_layer_name=None,
                                   use_histogram=False,
                                   *args,
                                   **kwargs):
        """Get the quantized variables.
        Use if to choose or skip the target should be quantized.
        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            quantize_first_convolution (boolean): True if using quantization at the first layer, False otherwise
            quantize_last_convolution (boolean): True if using quantization at the last layer, False otherwise
            first_layer_name (string): prefix of name of the weight nodes in the first layer
            last_layer_name (string): prefix of name of the weight nodes in the last layer
            use_histogram (boolean): True to return tf.summary.histogram of quantized var
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                if quantize_first_convolution is not None and first_layer_name is not None:
                    if not quantize_first_convolution and var.op.name.startswith(first_layer_name):
                        return var

                if quantize_last_convolution is not None and last_layer_name is not None:
                    if not quantize_last_convolution and var.op.name.startswith(last_layer_name):
                        return var

                if use_histogram:
                    return tf.summary.histogram("quantized_kernel", weight_quantization(var))
                else:
                    return weight_quantization(var)

        return var
