Weight quantization
===================

Blueoil can quantize weight in the network by passing the callable ``weight quantizer`` and keyword arguments ``weight_quantizer_kwargs`` to the network class.

.. literalinclude:: ../../blueoil/networks/classification/quantize_example.py
   :language: python
   :lines: 46-59, 70-79
   :emphasize-lines: 10-11, 20,23

Tensorflow custom getter
------------------------

The main idea of selecting variable to do weight quantize in Blueoil is from the ``custom_getter`` in ``variable_scope`` namely ``_quantized_variable_getter``.

.. literalinclude:: ../../blueoil/networks/classification/quantize_example.py
   :language: python
   :lines: 82-89, 103-
   :emphasize-lines: 1, 11-25, 30-36

The selection criteria is based on these three variables.

    - ``name``: This is an argument for ``tf.compat.v1.variable_scope(name)`` which indicate the raw_ops in this layer.
    - ``quantize_first_convolution``: boolean indicate quantization on the **first** convolution layer
    - ``quantize_last_convolution``: boolean indicate quantization on the **last** convolution layer

The variable which variable scope name ending with ``kernel`` will be weight quantized, except it is a first or last layer with ``quantize_first_convolution`` or ``quantize_last_convolution`` set as `False` respectively.

Weight quantizer
----------------

Selection of weight quantizer are ``Binary channel wise mean scaling quantizer`` and ``Binary mean scaling quantizer``:

Binary channel wise mean scaling quantizer (``BinaryChannelWiseMeanScalingQuantizer``)
______________________________________________________________________________________

This quantization creates a binary channel wise mean scaling quantizer.
If ``backward`` is provided, this ``backward`` will be used in backpropagation.

This method is varient of XNOR-Net [1]_ weight quantization, the differencce from XNOR-Net is backward function.

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
______________________________________________________________

This quantization creates a binary mean scaling quantizer.
If ``backward`` is provided, this ``backward`` will be used in backpropagation.

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

Reference

    .. [1] `XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks <https://arxiv.org/abs/1603.05279>`_
    .. [2] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_
