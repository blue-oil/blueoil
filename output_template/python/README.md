# Getting Started
```
$ pip install -r requirements.txt
$ python run.py -i [your image file] -c ../models/meta.yaml -m ../models/lib/lib_x86.so
```
## Prerequisites
```
Python = 2.7, 3.5 and 3.6
pip >= 9.0.1
```


# Demo

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

# Demo on GPU (Jetson)

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

### Tensorflow
The project is developed and tested in TensorFlow v1.4.1.
Please ensure specific tensorflow `CPU/GPU` version is installed.

> _tensorflow-gpu==1.4.1_ has been used

### CUDA
Cuda requirement highly depends on TensorFlow version.
For TF 1.4.1 we need Cuda-toolkit 8.0 with Cudnn 6.0.
For installation details click [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
