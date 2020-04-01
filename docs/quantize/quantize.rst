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

Tensorflow custom getter
########################


Kernel quantization
###################


Activation function to feature map
##################################


Quantized network class template
################################
.. literalinclude:: quantize_example.py
   :language: python
   :emphasize-lines: 77,79,103-104

Reference
*********
        .. [1] `XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks <https://arxiv.org/abs/1603.05279>`_
        .. [2] `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_
        - `Deep Learning with Low Precision by Half-wave Gaussian Quantization <https://arxiv.org/abs/1702.00953>`_
