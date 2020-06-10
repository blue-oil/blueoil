# LmnetV1Quantize

* Image size: {height: 32, width: 32}
* Number of class: 10

| Name | Param | Size (MB) | 1 bits Quant Size (MB) | FLOPs (m) |
| :-- | --: | --: | --: | --: |
| **total** | **717226** | **2.736** | **0.08908** | **267.08005** |
| **conv1** | **928** | **0.00354** | **0.00035** | **1.76947** |
| &nbsp;&nbsp;conv1/BatchNorm | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv1/BatchNorm/beta | 32 | 0.00012 | 0.00012 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv1/BatchNorm/gamma | 32 | 0.00012 | 0.00012 | - |
| &nbsp;&nbsp;conv1/conv2d | 864 | 0.0033 | 0.0001 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv1/conv2d/kernel | 864 | 0.0033 | 0.0001 | - |
| **conv2** | **18560** | **0.0708** | **0.00269** | **37.74874** |
| &nbsp;&nbsp;conv2/BatchNorm | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv2/BatchNorm/beta | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv2/BatchNorm/gamma | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;conv2/conv2d | 18432 | 0.07031 | 0.0022 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv2/conv2d/kernel | 18432 | 0.07031 | 0.0022 | - |
| **conv3** | **295168** | **1.12598** | **0.03613** | **150.99494** |
| &nbsp;&nbsp;conv3/BatchNorm | 256 | 0.00098 | 0.00098 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv3/BatchNorm/beta | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv3/BatchNorm/gamma | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;conv3/conv2d | 294912 | 1.125 | 0.03516 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv3/conv2d/kernel | 294912 | 1.125 | 0.03516 | - |
| **conv4** | **73856** | **0.28174** | **0.00928** | **37.74874** |
| &nbsp;&nbsp;conv4/BatchNorm | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv4/BatchNorm/beta | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv4/BatchNorm/gamma | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;conv4/conv2d | 73728 | 0.28125 | 0.00879 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv4/conv2d/kernel | 73728 | 0.28125 | 0.00879 | - |
| **conv5** | **295168** | **1.12598** | **0.03613** | **37.74874** |
| &nbsp;&nbsp;conv5/BatchNorm | 256 | 0.00098 | 0.00098 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv5/BatchNorm/beta | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv5/BatchNorm/gamma | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;conv5/conv2d | 294912 | 1.125 | 0.03516 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv5/conv2d/kernel | 294912 | 1.125 | 0.03516 | - |
| **conv6** | **32896** | **0.12549** | **0.00439** | **1.04858** |
| &nbsp;&nbsp;conv6/BatchNorm | 128 | 0.00049 | 0.00049 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv6/BatchNorm/beta | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv6/BatchNorm/gamma | 64 | 0.00024 | 0.00024 | - |
| &nbsp;&nbsp;conv6/conv2d | 32768 | 0.125 | 0.00391 | - |
| &nbsp;&nbsp;&nbsp;&nbsp;conv6/conv2d/kernel | 32768 | 0.125 | 0.00391 | - |
| **conv7** | **650** | **0.00248** | **0.00011** | **0.02064** |
| &nbsp;&nbsp;conv7/bias | 10 | 4e-05 | 4e-05 | - |
| &nbsp;&nbsp;conv7/kernel | 640 | 0.00244 | 8e-05 | - |
| **pool7** | **-** | **-** | **-** | **0.00016** |
| **Softmax** | **-** | **-** | **-** | **5e-05** |

