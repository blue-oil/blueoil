# Getting Started
```
$ pip install -r requirements.txt
$ python run.py -i [your image file] -c ../models/meta.yaml -l ../models/lib/lib_x86.so
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
$ python usb_camera_demo.py -c ../models/meta.yaml -l ../models/lib/lib_x86.so
```
