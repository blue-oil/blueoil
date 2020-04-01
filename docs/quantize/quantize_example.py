from functools import partial

import tensorflow as tf

from blueoil.networks.<TASK_NAME>.<NETWORK_NAME> import FooNetwork


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
