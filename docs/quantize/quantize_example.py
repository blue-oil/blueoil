from functools import partial

import tensorflow as tf

from blueoil.blocks import darknet
from blueoil.networks.classification.base import Base
from blueoil.layers import conv2d

class FooNetwork(Base):
    """Example model with simple layer"""

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")
        self.before_last_activation = self.activation

    def base(self, images, is_training):
        if self.data_format == "NCHW":
            channel_data_format = "channels_first"
        elif self.data_format == "NHWC":
            channel_data_format = "channels_last"
        else:
            raise RuntimeError("data format {} should be in ['NCHW', 'NHWC]'.".format(self.data_format))

        self.inputs = self.images = images

        darknet_block = partial(darknet, is_training=is_training,
                                activation=self.activation, data_format=self.data_format)

        x = darknet_block("block_1", self.inputs, filters=32, kernel_size=1)
        x = darknet_block("block_2", x, filters=8, kernel_size=3)
        x = self._reorg("pool_1", x, stride=2, data_format=self.data_format)

        output_filters = (self.num_classes + 5) * self.boxes_per_cell
        self.block_last = conv2d("block_last", x, filters=output_filters, kernel_size=1,
                                 activation=None, use_bias=True, is_debug=self.is_debug,
                                 data_format=self.data_format)

        return self.base_output

class FooNetworkQuantize(FooNetwork):
    """Quantize Foo Network."""

    def __init__(
            self,
            quantize_first_convolution=True,
            quantize_last_convolution=True,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        """
        Args:
            quantize_first_convolution(bool): use quantization in first conv.
            quantize_last_convolution(bool): use quantization in last conv.
            weight_quantizer (callable): weight quantizer.
            weight_quantize_kwargs(dict): Initialize kwargs for weight quantizer.
            activation_quantizer (callable): activation quantizer
            activation_quantize_kwargs(dict): Initialize kwargs for activation quantizer.
        """

        super().__init__(
            *args,
            **kwargs,
        )

        self.quantize_first_convolution = quantize_first_convolution
        self.quantize_last_convolution = quantize_last_convolution

        activation_quantizer_kwargs = activation_quantizer_kwargs if not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if not None else {}

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        self.weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

        if self.quantize_last_convolution:
            self.before_last_activation = self.activation
        else:
            self.before_last_activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

    @staticmethod
    def _quantized_variable_getter(
            weight_quantization,
            quantize_first_convolution,
            quantize_last_convolution,
            getter,
            name,
            *args,
            **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            weight_quantization: Callable object which quantize variable.
            quantize_first_convolution(bool): Use quantization in first conv.
            quantize_last_convolution(bool): Use quantization in last conv.
            getter: Default from tensorflow.
            name: Default from tensorflow.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.compat.v1.variable_scope(name):
            if "kernel" == var.op.name.split("/")[-1]:

                if not quantize_first_convolution:
                    if var.op.name.startswith("block_1/"):
                        return var

                if not quantize_last_convolution:
                    if var.op.name.startswith("block_last/"):
                        return var

                # Apply weight quantize to variable whose last word of name is "kernel".
                quantized_kernel = weight_quantization(var)
                tf.compat.v1.summary.histogram("quantized_kernel", quantized_kernel)
                return quantized_kernel

        return var

    def base(self, images, is_training):
        custom_getter = partial(
            self._quantized_variable_getter,
            self.weight_quantization,
            self.quantize_first_convolution,
            self.quantize_last_convolution,
        )
        with tf.compat.v1.variable_scope("", custom_getter=custom_getter):
            return super().base(images, is_training)
