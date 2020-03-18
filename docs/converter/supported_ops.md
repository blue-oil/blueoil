# Supported Ops
## Ops with Limitations
### Base Limitations
- **Output**
    - Requires each output channel size <= `1024`.

### Blueoil Customized Ops
- **[QTZ_binary_channel_wise_mean_scaling](https://github.com/blue-oil/blueoil/blob/620ba3b404dea142ff53461206c31e987b26cb6e/blueoil/converter/core/operators.py#L2352)**: Quantization operator using binary channel wise scaling.
- **[QTZ_binary_mean_scaling](https://github.com/blue-oil/blueoil/blob/620ba3b404dea142ff53461206c31e987b26cb6e/blueoil/converter/core/operators.py#L709)**: Quantization operator using binary scaling.
    - Input tensor must have float values.
- **[QTZ_linear_mid_tread_half](https://github.com/blue-oil/blueoil/blob/620ba3b404dea142ff53461206c31e987b26cb6e/blueoil/converter/core/operators.py#L1373)**: Quantization operator with 'linear mid tread half'.

### Tensorflow Ops with Limitations
- **[tf.layers.AveragePooling2D](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/AveragePooling2D)**
    - Currently, support only `2D`.
    - Do ***not*** support `kernel depth = 1`.
- **[tf.concat](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/concat)**
    - Do ***not*** support concat of mixed data types (e.g., quantized values and float values).
    - All tensor channels must be equal. 
    - If inputs are quantized, requires `Each input channel size = multiple of 32`.
- **[tf.layers.Conv2D](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/Conv2D)**
    - Support only convolution `2D`.
    - Do ***not*** support transpose.
    - Requires `kernel size = 1x1` or `3x3` or `5x5`.
        - Accelerator is not supported `kernel size = 5x5` (CPU supported only).
    - Requires `Input channel size = multiple of 32`, otherwise zero padding is used.
    - If output is quantized by later operations, `Output channel size = multiple of 32`, otherwise output channel size is free from limitation (but performance will be worse).
- **[tf.nn.depth_to_space](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/depth_to_space)**
    - Requires `depth of input = multiple of block_size^2 * 32`.
- **[tf.nn.fused_batch_norm](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/fused_batch_norm)**
    - `scale`, `offset`, `mean`, `variance`, and `epsilon` must be constants or computable from constants.
- **[tf.linalg.matmul](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/linalg/matmul)**
    - Do ***not*** support `scalar`.
- **[tf.layers.max_pooling2d](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/max_pooling2d)**
     - Currently, support only `2D`.
- **[tf.pad](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/pad)**
    - Supports only `channel-wise paddings`.
- **[tf.nn.space_to_depth](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/space_to_depth)**
    - Requires `output depth = (multiple of block_size^2 * 32)` or `(block_size^2 * {8, 16})`.
- **[tf.split](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/split)**
    - Currently, all of output tensors must have `same` shape.
    - For quantized tensor, requires `number of channel of each output tensor = multiple of 32`.

###  Tensorflow Ops without Limitations
- **[tf.math.add](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/add)**
- **[tf.identity](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/identity)**
- **[tf.nn.leaky_relu](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/leaky_relu)**
- **[tf.math.maximum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/maximum)**
- **[tf.math.minimum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/minimum)**
- **[tf.math.multiply](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/multiply)**
- **[tf.nn.relu](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/relu)**
- **[tf.reshape](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/reshape)**
- **[tf.image.resize_nearest_neighbor](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/image/resize_nearest_neighbor)**
- **[tf.nn.softmax](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/softmax)**
- **[tf.transpose](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/transpose)**


## Data Types
- **Floating point**
    - [tf.float32](https://www.tensorflow.org/api_docs/python/tf#float32): 32-bit single-precision floating-point.
