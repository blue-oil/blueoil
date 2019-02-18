# Summary

- [Getting Started with FPGA](#getting-started-(FPGA-Board))
    - [Prerequisites](#prerequisites)
    - [Camera Demo](#camera-demo-(fpga))
- [Getting Started with GPU](#getting-started-(gpu))
    - [Prerequisites](#prerequisites)
    - [Camera Demo](#camera-demo-(gpu))


# Getting Started (FGPA Board)

Please check the whole documentation detail on how to setup this demo on FPGA board [here](https://docs.blue-oil.org/install/install.html#setup-an-fpga-board)

```
$ pip install -r requirements.txt
$ python run.py -i [your image file] -c ../models/meta.yaml -m ../models/lib/lib_fpga.so
```
## Prerequisites
```
Python = 2.7, 3.5 and 3.6
pip >= 9.0.1
```

# Camera Demo (FPGA)

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
$ python usb_camera_demo.py -c ../models/meta.yaml -m ../models/lib/lib_x86.so
```

If you want to check more in detail about how to run the demonstration on FPGA board, please check the online [documentation](https://docs.blue-oil.org/tutorial/run_fpga.html#run-the-demonstration).


# Getting Started (GPU)

## Prerequisites
```
Python = 3.5
pip >= 9.0.1
```

### Tensorflow
The project is developed and tested in TensorFlow v1.4.1.
Please ensure specific tensorflow `GPU` version is installed.

> _tensorflow-gpu==1.4.1_ has been used

```
$ pip install tensorflow-gpu=1.4.1
```

### CUDA
Cuda requirement highly depends on TensorFlow version.
For TF 1.4.1 we need Cuda-toolkit 8.0 with Cudnn 6.0.
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
Python = 3.5
Opencv = 3.1.0
python opencv = 3.1.0
```

```
$ pkg-config --modversion opencv
3.1.0

$ python3 -c "import cv2; print(cv2.__version__)"
3.1.0
```