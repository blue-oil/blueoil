# Supported Ops
## Converter Ops with Limitations
### Base Limitations
- **Output**
    - Requires each output channel size <= `1024`

### Blueoil Customized Ops
- **QTZ_binary_channel_wise_mean_scaling**
- **QTZ_binary_mean_scaling**
- **QTZ_linear_mid_tread_half**

### Tensorflow Supported with Limitations
- **[tf.concat](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/concat)**
    - Do not support concat of mixed data types (e.g., quantized values and float values)
- **[tf.layers.Conv2D](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/Conv2D)**
    - Support only convolution `2D`
    - Requires kernel size = `1x1` or `3x3`
    - Requires Input/output channel size = `multiple of 32`, otherwise zero padding is used
    - Do not support transpose 
- **[tf.nn.fused_batch_norm](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/fused_batch_norm)**
    - Batch normalization just before quantization is folded with quantization.
    - Currently this function is only used by threshold skipping optimization pass for recursively calculating thresholds of the skipping patterns.
- **[tf.layers.AveragePooling2D](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/AveragePooling2D)**
    - Currently, support only `2D`
- **[tf.nn.depth_to_space](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/depth_to_space)**
    - Requires depth of input = `multiple of kernel_size^2 * 32`
- **[tf.linalg.matmul](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/linalg/matmul)**
    - Do not support scalar (matrix size = [a,`1`] or [`1`,a])
- **[tf.layers.max_pooling2d](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/max_pooling2d)**
     - Currently, support only `2D`
- **[tf.pad](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/pad)**
    - Supports only `channel-wise paddings`
- **[tf.nn.space_to_depth](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/space_to_depth)**
    - Requires output depth = `(multiple of kernel_size^2 * 32)` or `(kernel_size^2 * {8, 16})`
- **[tf.split](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/split)**
    - Currently, all of output tensors must have `same` shape
    - For quantized tensor, requires number of channel of each output tensor = `multiple of 32`

###  Tensorflow Supported without Limitations
- **[tf.math.add](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/add)**
- **[tf.cast](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/cast)**
- **[tf.gather](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/gather)**
- **[tf.identity](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/identity)**
- **[tf.nn.leaky_relu](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/leaky_relu)**
- **[tf.math.maximum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/maximum)**
- **[tf.math.minimum](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/minimum)**
- **[tf.math.multiply](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/multiply)**
- **[tf.keras.backend.prod](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/backend/prod)**
- **[tf.nn.relu](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/relu)**
- **[tf.reshape](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/reshape)**
- **[tf.shape](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/shape)**
- **[tf.nn.softmax](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/softmax)**
- **[tf.strided_slice](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/strided_slice)**
- **[tf.transpose](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/transpose)**
- **[tf.unique](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/unique)**

 ### Not supported operators 
 TensorFlow importer cannot convert these operator node.
 - **[tf.layers.batch_normalization](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/batch_normalization)**
 - **[tf.layers.conv2d_transpose](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/conv2d_transpose)**
 - **[tf.layers.Dropout](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/Dropout)**
 - **[tf.layers.Flatten](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/Flatten)**
  - **Gemm**

## Converter Data Types
### Supported Data Types
- **Floating point**
    - [tf.float32](https://www.tensorflow.org/api_docs/python/tf#float32): 32-bit single-precision floating-point.
    - [tf.float64](https://www.tensorflow.org/api_docs/python/tf#float64): 64-bit double-precision floating-point.
- **Integer**
    - [tf.int8](https://www.tensorflow.org/api_docs/python/tf#int8): 8-bit signed integer.
    - [tf.int16](https://www.tensorflow.org/api_docs/python/tf#int16): 16-bit signed integer.
    - [tf.int32](https://www.tensorflow.org/api_docs/python/tf#int32): 32-bit signed integer.
    - [tf.int64](https://www.tensorflow.org/api_docs/python/tf#int64): 64-bit signed integer.
- **Unsigned integers**
    - [tf.uint8](https://www.tensorflow.org/api_docs/python/tf#uint8): 8-bit unsigned integer.
    - [tf.uint16](https://www.tensorflow.org/api_docs/python/tf#uint16): 16-bit unsigned integer.
    - [tf.uint32](https://www.tensorflow.org/api_docs/python/tf#uint32): 32-bit unsigned integer.
    - [tf.uint64](https://www.tensorflow.org/api_docs/python/tf#uint64): 64-bit unsigned integer.
- **String**
    - [tf.string](https://www.tensorflow.org/api_docs/python/tf#string): String.
- **Boolean**
    - [tf.bool](https://www.tensorflow.org/api_docs/python/tf#bool): Boolean
- **DT_INVALID**

### Unsupported Data Types
- **Complex Numbers**
    - [tf.complex64](https://www.tensorflow.org/api_docs/python/tf#complex64): 64-bit single-precision complex.
    - [tf.complex128](https://www.tensorflow.org/api_docs/python/tf#complex128): 128-bit single-precision complex.

- **Custom**
    - **DT_HALF**
- **Primitive vectors**
    - **FLOATS**
    - **INTS**
- **Custom vectors**
    - **STRINGS**
- **Struct**
    - **TENSOR, t**
    - **GRAPH, g**
- **Struct vectors**
    - **TENSORS**
    - **GRAPHS**
