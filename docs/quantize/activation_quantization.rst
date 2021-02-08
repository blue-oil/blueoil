Activation quantization
=======================

Blueoil can quantize an activation function by passing the callable ``activation quantizer`` and keyword arguments ``activation_quantizer_kwargs`` to the network class.

.. literalinclude:: ../../blueoil/networks/classification/quantize_example.py
   :language: python
   :lines: 46-59, 70-79
   :emphasize-lines: 8-9, 21,24

Activation quantizer
--------------------

Currenly, Blueoil has only one activation function quantizer.

Linear mid tread half quantizer (``LinearMidTreadHalfQuantizer``)
_________________________________________________________________

This quantization creates a linear mid tread half quantizer.
If ``backward`` is provided, this ``backward`` will be used in backpropagation.

This quantization method is DoReFa-Net [1]_ activation quantization variant, the difference from DoReFa-Net is to be able to change ``max_value``.

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


Reference

    - `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_
    .. [1] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_
