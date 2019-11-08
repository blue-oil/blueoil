# Training for Semantic Segmentation

This guide covers training a neural network model on a GPU server to perferm semantic segmentation of on the [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset.

<img src="../_static/camvid.jpg" width="600">

## Preparation

The CamVid dataset is available from the SegNet tutorial on [Github](https://github.com/alexgkendall/SegNet-Tutorial), but the dataset contains fixed absolute paths of the data like [`/SegNet/CamVid/test/0001TP_008550.png`](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/CamVid/test.txt). We have changed the absolute paths to relative paths. You can download out modified dataset from our mirror site.

```
$ wget https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/camvid.tgz
$ tar -xzf camvid.tgz
```

The directory structure of this dataset is shown below. In the CamVid dataset, both training and annotation data are binary image files.

```
CamVid
 ├─ label_colors.txt
 │
 ├─ train.txt
 ├─ train
 │   ├─ 0001TP_006690.png
 │   ├─ 0001TP_007500.png
 │   └─ ...
 ├─ trainannot
 │   ├─ 0001TP_006690.png
 │   ├─ 0001TP_007500.png
 │   └─ ...
 │
 ├─ val.txt
 ├─ val
 │   ├─ 0016E5_07959.png
 │   ├─ 0016E5_07975.png
 │   └─ ...
 ├─ valannot
 │   ├─ 0016E5_07959.png
 │   ├─ 0016E5_07975.png
 │   └─ ...
 │
 ├─ test.txt
 ├─ test
 │   ├─ 0001TP_008550.png
 │   ├─ 0001TP_008910.png
 │   └─ ...
 └─ testannot
     ├─ 0001TP_008550.png
     ├─ 0001TP_008910.png
     └─ ...
```

CamVid dataset consists of 360x480 color images in 12 classes. There are 367 training images, 101 validaton images and 233 test images.

## Generate a configuration file

Generate your model configuration file interactively by running the `python blueoil/cmd/main.py init` command.

    $ PYTHONPATH=.:lmnet:dlk/python/dlk python blueoil/cmd/main.py init

This is an example of the initialization procedure.

```
#### Generate config ####
your model name ():  camvid
choose task type  semantic_segmentation
choose network  LmSegnetV1Quantize
choose dataset format  CamvidCustom
training dataset path:  {dataset_dir}
set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  yes
test dataset path:  {dataset_dir}
batch size (integer):  8
image size (integer x integer):  360x480
how many epochs do you run training (integer):  1000
choose optimizer: Adam
initial learning rate:  0.001
choose learning rate schedule ({epochs} is the number of training epochs you entered before):  '3-step-decay' -> learning rate decrease by 1/10 on {epochs}/3 and {epochs}*2/3 and {epochs}-1
enable data augmentation?  Yes
Please choose augmentors:  done (5 selections)
-> select Brightness, Color, Contrast, FlipLeftRight, Hue
apply quantization at the first layer?  no
```

If configuration finishes, the configuration file is generated in the `{Model name}.yml` under current directory.

When you wnat to create config yaml in specific filename or directory, you can use `-o` option.

    $ PYTHONPATH=.:lmnet:dlk/python/dlk python blueoil/cmd/main.py init -o ./configs/my_config.yml

## Train a network model

Train your model by running `python blueoil/cmd/main.py train` command with model configuration.

    $ PYTHONPATH=.:lmnet:dlk/python/dlk python blueoil/cmd/main.py train -c {PATH_TO_CONFIG.yml}

When training has started, the training log and checkpoints will be generated under `./saved/{Mode name}_{TIMESTAMP}`.

Training is running on the TensorFlow backend. So you can use TensorBoard to visualize your training progress.

    $ tensorboard --logdir=saved/{Model name}_{TIMESTAMP} --port {Port}

- Learning Rate / Loss
<img src="../_static/semantic_segmentation_loss.png">

- Metrics / IOU
<img src="../_static/semantic_segmentation_iou.png">

- Images / Overlap Output Input
<img src="../_static/semantic_segmentation_overwrap.png">

## Convert training result to FPGA ready format.

Convert trained model to executable binary files for x86, ARM, and FPGA.
Currently, conversion for FPGA only supports Intel Cyclone® V SoC FPGA.

    $ PYTHONPATH=.:lmnet:dlk/python/dlk python blueoil/cmd/main.py convert -e {Model name}

`python blueoil/cmd/main.py convert` automatically executes some conversion processes.
- Convert Tensorflow checkpoint to protocol buffer graph.
- Optimize graph
- Generate source code for executable binary
- Compile for x86, ARM and FPGA

If conversion is successful, output files are generated under `./saved/{Mode name}_{TIMESTAMP}/export/save.ckpt-{Checkpoint No.}/{Image size}/output`.

```
output
 ├── fpga (include preloader and FPGA configuration file)
 │   ├── preloader-mkpimage.bin
 │   ├── soc_system.rbf
 │   └── soc_system.dtb
 ├── models
 │   ├── lib (include trained model library)
 │   │   ├── lib_arm.so
 │   │   ├── lib_fpga.so
 │   │   └── lib_x86.so
 │   └── meta.yaml (model configuration)
 ├── python
 │   ├── lmnet (include pre-process/post-process)
 │   ├── README.md
 │   ├── requirements.txt
 │   ├── run.py (inference script)
 │   └── usb_camera_demo.py (demo script for object detection and classification)
 └── README.md
```

## Run inference script on x86 Linux (Ubuntu 16.04)

- Prepare images for inference

	You can find test imgaes on CamVid's [Official Site](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

		$ wget http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/pr/0006R0_f02040.jpg

- Run inference script

    Explore into the `output/python` directory, and
    run `run.py`. Inference results are saved in `./output/images/{image_name}.png`.

    Note: If you run the script for the first time, you have to setup a python environment (2.7 or 3.5+) and required python packages.

	```
	$ cd {output/python directory}
	$ sudo pip install -r requirements.txt  # for the first time only
	$ python run.py \
	      -i {inference image path} \
	      -m ../models/lib/lib_x86.so \
	      -c ../models/meta.yaml
	```

- Check inference results

	Image will be exported to `output/images/{image_name}.png`. Below you can see an example of the expected output.

	<img src="../_static/semantic_segmentation_output.png">
