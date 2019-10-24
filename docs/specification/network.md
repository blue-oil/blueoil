## Network model's name, version and structure

Policy of network name, version and structure.


Source code structure:
```
lmnet/networks/
├── classification
│   ├── base.py
│   ├── darknet.py
│   ├── lmnet_v0.py
│   ├── lmnet_v1.py
│   ├── resnet.py
│   └── vgg16.py
├── object_detection
│   ├── ssd.py
│   ├── yolo_v1.py
│   └── yolo_v2.py
└── segmentation
    ├── base.py
    └── lm_segnet.py
```


Name and version: 
* Name of `LMNet` or `Lmnet` to be used for classification network. 
* Our original model should start with `lm` prefix.  
  caution for working members: **We will rename old classification `lmnet` to `lmnet_v0`.**
* For network versioning, we will use postfix of `v`.
* Technically, the network **class name** should be camel case, and **file name** should be in all lower-case.  
  ex.
  * class: YoloV2, file: yolo_v2.py
  * class: LmnetV1, file: lmnet_v1.py
*  The **quantized** version of same network should be in the **same file** as non-quantized network. However if the file becomes too large, another file may be used.
* When we needs create some variations of the same network (LMNet), I will use `Boxing weight class` or `Font weight class`, It is  good metaphor for represent model's weights. It isn't decided yet. 
ex. LMNetHeavy, LMNetWelter.
