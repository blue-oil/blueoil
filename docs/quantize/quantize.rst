************
Quantization
************

Weight quantization
###################
Binary channel wise mean scaling quantizer (``BinaryChannelWiseMeanScalingQuantizer``)
**************************************************************************************
        This quantization creates a binary channel wise mean scaling quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This method is varient of XNOR-Net [1]_ weight quantization, the differencce from XNOR-Net [1]_ is backward function.

        Forward is:

        .. math::
            \begin{align}
                \bar{\mathbf{x}} & = \frac{1}{n}||\mathbf{X}||_{\ell1}
                & \text{$\bar{\mathbf{x}}$ is a $c$-channels vector} \\
                & & \text{$n$ is number of elements in each channel of $\mathbf{X}$} \\\\
                \mathbf{Y} & = \text{sign}\big(\mathbf{X}\big) \times \bar{\mathbf{x}} &\\
            \end{align}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} = \frac{\partial Loss}{\partial \mathbf{Y}}

Binary mean scaling quantizer (``BinaryMeanScalingQuantizer``)
**************************************************************
        This quantization creates a binary mean scaling quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This method is DoReFa-Net [2]_ weight quantization.

        Forward is:

        .. math::
            \begin{align}
                \bar{x} & = \frac{1}{N}||\mathbf{X}||_{\ell1}
                & \text{$\bar{x}$ is a scalar} \\
                & & \text{$N$ is number of elements in all channels of $\mathbf{X}$}\\
                \mathbf{Y} & = \text{sign}\big(\mathbf{X}\big) \cdot \bar{x} &\\
            \end{align}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} = \frac{\partial Loss}{\partial \mathbf{Y}}


Activation quantization
#######################
Linear mid tread half quantizer (``LinearMidTreadHalfQuantizer``)
*****************************************************************
        This quantization creates a linear mid tread half quantizer.
        If `backward` is provided, this `backward` will be used in backpropagation.

        This quantization method is DoReFa-Net [2]_ activation quantization variant, the differencce from DoReFa-Net [2]_ is to be able to change `max_value`.

        Forward is:

        .. math::
            \mathbf{X} & = \text{clip}\big(\mathbf{X}, 0, max\_value\big)\\
            \mathbf{Y} & =
                \begin{cases}
                \mathbf{X},  & \text{if $bit$ is 32} \\
                \frac{\text{round}\big(\frac{\mathbf{X}}{max\_value}
                    \cdot (2^{bit}-1)\big)}{2^{bit}-1} \cdot max\_value, & otherwise
                \end{cases}

        Default backward is:

        .. math::
            \frac{\partial Loss}{\partial \mathbf{X}} =
                \begin{cases}
                \frac{\partial Loss}{\partial y},  & \text{if $0 < x < max\_value$}\\
                0, & otherwise
                \end{cases}


Kernel quantization
###################


Activation function to feature map
##################################


Quantized network class template
################################
.. code-block:: python

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
                raise RuntimeError("data format {} shodul be in ['NCHW', 'NHWC]'.".format(self.data_format))

            self.inputs = self.images = images

            darknet_block = partial(darknet, is_training=is_training,
                                    activation=self.activation, data_format=self.data_format)

            x = darknet_block("block_1", self.inputs, filters=32, kernel_size=1)
            x = darknet_block("block_2", x, filters=8, kernel_size=3)
            x = self._reorg("pool_1", x, stride=2, data_format=self.data_format)

            output_filters = (self.num_classes + 5) * self.boxes_per_cell
            self.block_last = conv2d("block_last", x, filters=output_filters, kernel_size=1,
                                     activation=None, use_bias=True, is_debug=self.is_debug,
                                     data_format=channel_data_format)

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


Reference
*********
        .. [1] `XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks <https://arxiv.org/abs/1603.05279>`_
        .. [2] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_
        - `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_
