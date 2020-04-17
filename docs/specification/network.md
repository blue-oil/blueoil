## Network model's name, version and structure

Policy of network name, version and structure.


Source code structure:
```
blueoil/networks/
├── classification
│   ├── base.py
│   ├── darknet.py
│   ├── lm_resnet.py
│   ├── lmnet_v0.py
│   ├── lmnet_v1.py
│   ├── mobilenet_v2.py
│   ├── quantuze_examply.py
│   ├── resnet.py
│   └── vgg16.py
├── keypoint_detection
│   ├── base
│   └── lm_single_pose_v1.py
├── object_detection
│   ├── lm_yolo.py
│   ├── yolo_v1.py
│   ├── yolo_v2.py
│   └── yolo_v2_quantize.py 
└── segmentation
    ├── base.py
    └── lmnet_multi.py
```


Name and version: 
* Name of `Lm` or `LM` denotes **LeapMind**.
* For **LeapMind original model**, `Lm{network}` or `LM{network}` are used for network **class name**. These networks **file name** should have **`lm` prefix**.  
  ex.
  * class: LmResnet, file: lm_resnet.py
  * class: LMFYolo, file: lm_fyolo.py  
* For network versioning, we will use postfix of `v`.  
    caution for working members: **We will rename old classification `lmnet` to `lmnet_v0.`**
* Technically, the network **class name** should be camel case, and **file name** should be in all lower-case.  
  ex.
  * class: YoloV2, file: yolo_v2.py
  * class: LmnetV1, file: lmnet_v1.py  

*  The **quantized** version of same network should be in the **same file** as non-quantized network. However if the file becomes too large, another file may be used.
* When we needs create some variations of the same network (LMNet), I will use `Boxing weight class` or `Font weight class`, It is good metaphor for represent model's weights. It is not decided yet. 
ex. LMNetHeavy, LMNetWelter.


## Network specification
THe result of inference test on Terasic DE10-nano using `lm_fpga.elf` and `lm_arm.elf`.

### LmnetV1Quantize
- Inference Speed (FPGA) **6.769** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 7677,  sum:7.677ms
    TotalRunTime 6769,  sum:6.769ms
    ..Convolution 3753,71,  sum:3.824ms
    ....kn2row 3659,  sum:3.659ms
    ......kn2row-buf 6,  sum:0.006ms
    ......matrix_multiplication 462,428,422,417,  sum:1.729ms
    ........matrix_transpose (row_major) 25,24,20,19,  sum:0.088ms
    ......matrix_shift_add_f 583,431,425,424,  sum:1.863ms
    ....kn2row-1x1 63,  sum:0.063ms
    ......matrix_multiplication 50,  sum:0.05ms
    ..BatchNorm 375,18,  sum:0.393ms
    ..LinearMidTreadHalfQuantizer 219,  sum:0.219ms
    ....pack_input 48,  sum:0.048ms
    ..QuantizedConv2D 500,680,256,253,114,  sum:1.803ms
    ....Convert Tensor 55,29,16,12,13,  sum:0.125ms
    ....Sync UDMABuf Input 100,77,44,29,24,  sum:0.274ms
    ....Conv2D TCA 257,520,151,176,36,  sum:1.14ms
    ....Sync UDMABuf Output 59,32,23,18,20,  sum:0.152ms
    ..Memcpy 78,24,18,12,  sum:0.132ms
    ..ExtractImagePatches 60,14,14,  sum:0.088ms
    ..QuantizedConv2D_ApplyScalingFactor 35,  sum:0.035ms
    ..ReLu 17,  sum:0.017ms
    ..Add 11,  sum:0.011ms
    ..AveragePool 21,  sum:0.021ms
    ..SoftMax 116,  sum:0.116ms
    ```
- Inference Speed (Arm) **16.526** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 5976,  sum:5.976ms
    TotalRunTime 16526,  sum:16.526ms
    ..Convolution 3923,73,  sum:3.996ms
    ....kn2row 3826,  sum:3.826ms
    ......kn2row-buf 6,  sum:0.006ms
    ......matrix_multiplication 578,434,427,418,  sum:1.857ms
    ........matrix_transpose (row_major) 38,28,24,20,  sum:0.11ms
    ......matrix_shift_add_f 606,436,432,428,  sum:1.902ms
    ....kn2row-1x1 64,  sum:0.064ms
    ......matrix_multiplication 50,  sum:0.05ms
    ..BatchNorm 374,16,  sum:0.39ms
    ..LinearMidTreadHalfQuantizer 218,  sum:0.218ms
    ....pack_input 50,  sum:0.05ms
    ..QuantizedConv2D 2515,5689,1445,1617,182,  sum:11.448ms
    ....Convert Tensor 25,20,16,11,13,  sum:0.085ms
    ....Quantized Conv2D Tiling 2472,5654,1414,1594,156,  sum:11.29ms
    ..Memcpy 37,24,13,10,  sum:0.084ms
    ..ExtractImagePatches 60,15,14,  sum:0.089ms
    ..QuantizedConv2D_ApplyScalingFactor 25,  sum:0.025ms
    ..ReLu 17,  sum:0.017ms
    ..Add 10,  sum:0.01ms
    ..AveragePool 16,  sum:0.016ms
    ..SoftMax 123,  sum:0.123ms
    ```

### LMFYoloQuantize
- Inference Speed (FPGA) **59.011** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 50436,  sum:50.436ms
    TotalRunTime 59011,  sum:59.011ms
    ..Convolution 9605,606,  sum:10.211ms
    ....kn2row-1x1 9587,596,  sum:10.183ms
    ......matrix_multiplication 9571,585,  sum:10.156ms
    ........matrix_transpose (row_major) 15,  sum:0.015ms
    ..BatchNorm 18696,94,  sum:18.79ms
    ..LinearMidTreadHalfQuantizer 14432,  sum:14.432ms
    ....pack_input 2736,  sum:2.736ms
    ..QuantizedConv2D 5718,1447,728,602,538,389,227,109,175,258,1258,230,226,230,262,  sum:12.397ms
    ....Convert Tensor 881,122,42,23,19,12,9,10,20,28,17,9,8,10,9,  sum:1.219ms
    ....Sync UDMABuf Input 974,343,185,98,62,37,20,34,57,95,59,24,19,24,20,  sum:2.051ms
    ....Conv2D TCA 3126,791,417,414,404,301,161,30,49,62,1142,161,161,161,162,  sum:7.542ms
    ....Sync UDMABuf Output 698,160,59,45,28,18,18,16,27,54,21,16,18,16,53,  sum:1.247ms
    ..Memcpy 859,167,49,33,24,12,15,12,22,47,15,11,14,11,  sum:1.291ms
    ..ExtractImagePatches 794,232,64,41,19,7,22,41,  sum:1.22ms
    ..func_ConcatOnDepth 26,  sum:0.026ms
    ..QuantizedConv2D_ApplyScalingFactor 177,  sum:0.177ms
    ..LeakyReLu 202,  sum:0.202ms
    ..Add 29,  sum:0.029ms
    ```
- Inference Speed (Arm) **141.915** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 49056,  sum:49.056ms
    TotalRunTime 141915,  sum:141.915ms
    ..Convolution 9586,607,  sum:10.193ms
    ....kn2row-1x1 9565,593,  sum:10.158ms
    ......matrix_multiplication 9549,582,  sum:10.131ms
    ........matrix_transpose (row_major) 35,  sum:0.035ms
    ..BatchNorm 18502,95,  sum:18.597ms
    ..LinearMidTreadHalfQuantizer 14234,  sum:14.234ms
    ....pack_input 2781,  sum:2.781ms
    ..QuantizedConv2D 44505,11135,5749,8457,4364,2619,1338,374,752,1237,10235,1308,1314,1302,1302,  sum:95.991ms
    ....Convert Tensor 664,130,41,22,18,13,9,12,25,36,15,12,9,9,9,  sum:1.024ms
    ....Quantized Conv2D Tiling 43808,10984,5690,8422,4329,2593,1316,351,712,1189,10207,1283,1294,1281,1282,  sum:94.741ms
    ..Memcpy 678,120,37,21,19,15,12,10,16,23,20,10,10,9,  sum:1ms
    ..ExtractImagePatches 774,232,65,41,21,8,24,39,  sum:1.204ms
    ..func_ConcatOnDepth 27,  sum:0.027ms
    ..QuantizedConv2D_ApplyScalingFactor 157,  sum:0.157ms
    ..LeakyReLu 243,  sum:0.243ms
    ..Add 29,  sum:0.029ms
    ```

### LmSegnetV1Quantize
- Inference Speed (FPGA) **400.509** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 78330,  sum:78.33ms
    TotalRunTime 400509,  sum:400.509ms
    ..Convolution 12586,  sum:12.586ms
    ....kn2row-1x1 12564,  sum:12.564ms
    ......matrix_multiplication 12546,  sum:12.546ms
    ........matrix_transpose (row_major) 28,  sum:0.028ms
    ..BatchNorm 16155,  sum:16.155ms
    ..LinearMidTreadHalfQuantizer 54457,  sum:54.457ms
    ....pack_input 44910,  sum:44.91ms
    ..ExtractImagePatches 2957,914,535,  sum:4.406ms
    ..QuantizedConv2D 2593,3730,3731,11471,11512,11497,11488,6611,14269,18339,18043,  sum:113.284ms
    ....Convert Tensor 740,226,201,187,208,200,203,228,604,3267,2944,  sum:9.008ms
    ....Sync UDMABuf Input 863,522,528,528,531,538,526,531,862,2295,2352,  sum:10.076ms
    ....Conv2D TCA 451,2637,2639,10405,10408,10404,10404,5226,10421,10414,10414,  sum:83.823ms
    ....Sync UDMABuf Output 492,316,333,326,334,329,331,602,2348,2329,2305,  sum:10.045ms
    ..Memcpy 664,325,322,304,346,310,355,634,3166,2707,  sum:9.133ms
    ..DepthToSpace 3188,7023,28765,  sum:38.976ms
    ..linear_to_float 129916,  sum:129.916ms
    ```
 - Inference Speed (Arm) **1437.04** ms
    ```
    -------------------------------------------------------------
    Comparison: Default network test  succeeded!!!
    -------------------------------------------------------------
    TotalInitTime 89726,  sum:89.726ms
    TotalRunTime 1.43704e+06,  sum:1437.04ms
    ..Convolution 12599,  sum:12.599ms
    ....kn2row-1x1 12578,  sum:12.578ms
    ......matrix_multiplication 12562,  sum:12.562ms
    ........matrix_transpose (row_major) 44,  sum:0.044ms
    ..BatchNorm 16459,  sum:16.459ms
    ..LinearMidTreadHalfQuantizer 43441,  sum:43.441ms
    ....pack_input 33903,  sum:33.903ms
    ..ExtractImagePatches 2999,920,548,  sum:4.467ms
    ..QuantizedConv2D 13961,34212,34125,153796,153803,153764,153742,67980,133109,136513,135597,  sum:1170.6ms
    ....Convert Tensor 576,228,214,220,199,190,182,217,590,3048,2411,  sum:8.075ms
    ....Quantized Conv2D Tiling 13345,33959,33887,153554,153576,153553,153540,67743,132486,133440,133164,  sum:1162.25ms
    ..Memcpy 609,239,227,226,270,255,259,717,2508,2493,  sum:7.803ms
    ..DepthToSpace 3206,7139,29542,  sum:39.887ms
    ..linear_to_float 119963,  sum:119.963ms
    ```
