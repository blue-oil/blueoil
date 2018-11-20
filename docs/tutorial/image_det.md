# Training for Object Detection

This guide trains a neural network model to object detection of Human Face and Human Hand in the [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html) on GPU server.

<img src="../_static/openimages_v4.png" width="600">

## Preparation

The Open Images V4 dataset is available from official website, but original dataset is too large for the tutorial. We provide reduced dataset.

Blueoil supports 2 formats for object detection.

- OpenImagev4 format
- DeLTA-Mark format

Note: *Please see the detail in <a href="../usage/dataset.html">Prepare training dataset</a>*

You can download subset of Open Images V4 from
[our server](https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/openimages.tgz).


	$ wget https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/openimages.tgz
	$ tar xzf openimages.tgz


This dataset consists of 5,000 color images in 2 classes, with 2,500 images per class.

## Generate a configuration file

Generate your model configuration file interactively by running `blueoil init` command.

    $ ./blueoil.sh init

This is an example of configuration.

```
#### Generate config ####
your model name ():  objectdetection
choose task type  object_detection
choose network  LMFYoloQuantize
choose dataset format  OpenImagesV4
training dataset path:  {dataset_dir}
set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  no
batch size (integer):  16
image size (integer x integer):  128x128
how many epochs do you run training (integer):  100
initial learning rate:  0.001
choose learning rate setting(tune1 / tune2 / tune3 / fixed):  tune1 -> "2 times decay": tune1
apply quantization at the first layer?  yes
```

## Train a neural network

You can train same as <a href="./image_cls.html">Classification example</a>.

## Convert training result to FPGA ready format.

You can convert same as <a href="./image_cls.html">Classification example</a>.

## Run inference script on x86 Linux (Ubuntu 16.04)

- Prepare inference images (not included in the training dataset)

    [ for example: sitting_man ]

    <img src="../_static/sitting_man.jpg" width=128>
- Run inference script

    Explore into the `output/python` directory, and
    run `run.py` and inference result is saved in `./output/output.json`.

    Note: If you run the script for the first time, you have to setup a python environment (2.7 or 3.5+) and install requirements python packages.

```
$ cd {output/python directory}
$ sudo pip install -r requirements.txt  # only the first time
$ python run.py \
      -i {inference image path} \
      -l ../models/lib/lib_x86.so \
      -c ../models/meta.yaml
```