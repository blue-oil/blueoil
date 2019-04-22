# Summary

- [Getting Started with FPGA, x86 and ARM](#getting-started-(FPGA,-x86-and-ARM))
    - [Prerequisites](#prerequisites)
    - [Camera Demo](#camera-demo)
- [Getting Started with GPU](#getting-started-(gpu))
    - [Prerequisites](#prerequisites)
    - [Camera Demo](#camera-demo-(gpu))


# Getting Started (FPGA, x86 and ARM)


```
$ pip install -r requirements.txt
$ python run.py -i [your image file] -c ../models/meta.yaml -m ../models/lib/lib_fpga.so
```
## Prerequisites
```
Python = 2.7, 3.5 and 3.6
pip >= 9.0.1
```

# Camera Demo

## Prerequisites

```
Python = 2.7, 3.5 and 3.6
pip >= 9.0.1
Opencv = 3.1.0
python opencv = 2.4.9.1
```

```
$ pkg-config --modversion opencv
3.1.0

$ python -c "import cv2; print(cv2.__version__)"
2.4.9.1
```

## Run

```
$ pip install -r requirements.txt
$ python usb_camera_demo.py -c ../models/meta.yaml -m ../models/lib/lib_fpga.so
```


# Getting Started (GPU)

## Prerequisites
```
Python = 3.5, 3.6
pip >= 9.0.1
```

### Tensorflow
The project is developed and tested in TensorFlow v1.13.1.
Please ensure specific tensorflow `GPU` version is installed.

> _tensorflow-gpu==1.13.1_ has been used

```
$ pip install tensorflow-gpu==1.13.1
```

### CUDA
Cuda requirement highly depends on TensorFlow version.
For TF 1.13.1 we need Cuda 10.0 with Cudnn 7.0.
For installation details click [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).


```
$ pip install -r requirements.txt
$ python run.py -i [your image file] -c ../models/meta.yaml -m minimal_graph_with_shape.pb
```

# Camera Demo (GPU)
## Run

```
$ pip install -r requirements.txt
$ python usb_camera_demo.py -c ../models/meta.yaml -m minimal_graph_with_shape.pb
```

## Prerequisites
```
Python = 3.5, 3.6
Opencv >= 3.1.0
python opencv >= 3.1.0
```

```
$ pkg-config --modversion opencv
3.3.1

$ python3 -c "import cv2; print(cv2.__version__)"
3.3.1
```
