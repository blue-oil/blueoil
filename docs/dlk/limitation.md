# Limitation

## DLK Data Types
### Supported data type
 - **DT_INVALID**
 - **DT_DOUBLE**
 - **DT_FLOAT**
 - **DT_INT64**
 - **DT_INT32**
 - **DT_INT16**
 - **DT_INT8**
 - **DT_UINT64**
 - **DT_UINT32**
 - **DT_UINT16**
 - **DT_UINT8**
 - **DT_BOOL**
 - **DT_STRING**
 - **f**
 - **i**
 - **s**
 
### Unsupported data type
 - **FLOATS**
 - **INTS**
 - **DT_HALF**
 - **DT_COMPLEX64**
 - **DT_COMPLEX128**
 - **STRINGS**
 - **TENSOR**
 - **GRAPH**
 - **t**
 - **g**
 - **TENSORS**
 - **GRAPHS**
 
## DLK Operators with limitations
### Base class
 - **Input**
 - **Output**
    - Require each output channel size <= `1024`
 - **Constant**
 - **Lookup** 
### Tensorflow operators with name substitution

 - **Conv** (Conv2D)
    - Support only convolution `2D`
    - Requires kernel size = `1x1` or `3x3`
    - Requires Input/output channel size = `multiple of 32`, otherwise zero padding is used
    - Not support transpose 

 - **BatchNormalization** (FusedBatchNorm, FusedbatchNormV3)
    - Batch normalization just before quantization is folded with quantization.

 - **AveragePool** (AvgPool)
    - Currently, support only `2D`

 - **Add** (AddV2, BiasAdd)
    
 - **ConcatOnDepth** (ConcatV2)
 
 - **Gather** (GatherV2)
 
 - **BinaryChannelWiseMeanScalingQuantizer** (QTZ_binary_channel_wise_mean_scaling)
 
 - **BinaryMeanScalingQuantizer** (QTZ_binary_mean_scaling)
 
### Tensorflow operators
 - **Cast**
 
 - **DepthToSpace**
    - Requires depth of input = `multiple of kernel_size^2 * 32`
 
 - **Identity**
 
 - **LeakyRelu**
 
 - **MatMul**
    - Not support scalar
    
 - **Maximum**
 
 - **MaxPool**
     - Currently, support only `2D`
     
 - **Minimum**
 
 - **Mul**
 
 - **Pad**
    - Supports only `channel-wise paddings`
    
 - **Prod**
 
 - **QTZ_linear_mid_tread_half**
 
 - **Relu**
 
 - **Reshape**
 
 - **Shape**
 
 - **Softmax**
 
 - **SpaceToDepth**
    - Requires output depth = `(multiple of kernel_size^2 * 32)` or `(kernel_size^2 * {8, 16})`
    
 - **Split**
    - Currently, all of output tensors must have `same` shape
    - For quantized tensor, requires number of channel of each output tensor = `multiple of 32`
    
 - **StridedSlice**
 
 - **Transpose**
 
 - **Unique**
 
 ### Not supported operators
  - **Dropout**
 
 - **Flatten**
 
 - **Gemm**