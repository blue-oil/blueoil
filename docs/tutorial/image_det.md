# Training for Object Detection

This guide trains a neural network model to object detection of Human Face in the [Open Images Dataset V4](https://storage.googleapis.com/openimages/web/index.html) on GPU server.

<img src="../_static/openimages_v4.png" width="600">

## Preparation

The Open Images V4 dataset is available from official website, but original dataset is too large for the tutorial. So, We provide reduced dataset.

Blueoil supports 2 formats for object detection.

- OpenImagev4 format
- DeLTA-Mark format

Note: *Please see the detail in <a href="../usage/dataset.html">Prepare training dataset</a>*

You can download subset of Open Images V4 from
[our server](https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/openimages_face.tgz).


	$ wget https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/openimages_face.tgz
	$ tar xf openimages_face.tgz


This dataset consists of 2869 Human Face images and 5171 annotation boxes.

## Generate a configuration file

Generate your model configuration file interactively by running `blueoil.sh init` command.

    $ ./blueoil.sh init

This is an example of initialization.

```
#### Generate config ####
your model name ():  objectdetection
choose task type  object_detection
choose network  LMFYoloQuantize
choose dataset format  OpenImagesV4
training dataset path:  {dataset_dir}
set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  no
batch size (integer):  16
image size (integer x integer):  224x224
how many epochs do you run training (integer):  1000
initial learning rate:  0.001
choose learning rate schedule ({epochs} is the number of training epochs you entered before):  '3-step-decay' -> learning rate decrease by 1/10 on {epochs}/3 and {epochs}*2/3 and {epochs}-1
enable data augmentation?  Yes
Please choose augmentors:  done (5 selections)
-> select Brightness, Color, FlipLeftRight, Hue, SSDRandomCrop
apply quantization at the first layer?  no
```

## Train a network model

Train your model by running `blueoil.sh train` command with model configuration.

    $ ./blueoil.sh train config/{Model name}.yml

When training is started, training log and checkpoints are generated under `./saved/{Mode name}_{TIMESTAMP}` directory.

Training is running on TensorFlow backend. So you can use TensorBoard to visualize your training process. 

    $ ./blueoil.sh tensorboard saved/{Model name}_{TIMESTAMP} {Port}

- Metrics / Accuracy
<img src="../_static/object_detection_train_metrics.png">

- Loss, Weight Decay
<img src="../_static/object_detection_train_loss.png">

- Images / Final Detect Boxes
<img src="../_static/object_detection_boxes.png">


## Convert training result to FPGA ready format.

Convert trained model to executable binary files for x86, ARM, and FPGA.
Currently, conversion for FPGA only supports Intel Cyclone® V SoC FPGA.

    $ ./blueoil.sh convert config/{Model name}.yml saved/{Mode name}_{TIMESTAMP}

`blueoil.sh convert` automatically executes some conversion processes.
- Convert Tensorflow checkpoint to protocol buffer graph.
- Optimize graph
- Generate source code for executable binary
- Compile for x86, ARM and FPGA

If conversion is successful, output files are generated under `./saved/{Mode name}_{TIMESTAMP}/export/save.ckpt-{Checkpoint No.}/{Image size}/output`.

```
output
 ├── fpga (include preloader and FPGA configuration file)
 │   ├── preloader-mkpimage.bin
 │   └── soc_system.rbf
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
 │   └── usb_camera_demo.py (demo script for object detection)
 └── README.md
```

## Run inference script on x86 Linux (Ubuntu 16.04)

- Prepare images for inference (not included in the training dataset)

	You can find test imgaes on [Creative Commons](https://ccsearch.creativecommons.org/). [Sample](https://ccsearch.creativecommons.org/photos/ddfd33a6-140f-49a3-85b8-3bf58a877990)

		$ wget https://farm4.staticflickr.com/1172/1144309435_eff42ee683_o.jpg

- Run inference script

    Explore into the `output/python` directory, and
    run `run.py` and inference result is saved in `./output/output.json`.

    Note: If you run the script for the first time, you have to setup a python environment (2.7 or 3.5+) and required python packages.

	```
	$ cd {output/python directory}
	$ sudo pip install -r requirements.txt  # for the first time only
	$ python run.py \
	      -i {inference image path} \
	      -l ../models/lib/lib_x86.so \
	      -c ../models/meta.yaml
	```

	*Tips:* The default thredhold for object detection is `0.05`. If you find too many boxses when running demo, you can edit `meta.yml` and set threshold to 0.4 or 0.5 as below code.

	```
	ExcludeLowScoreBox:
	    threshold: 0.4
	```

- Check inference result

	```
	{
	    "classes": [
	        {
	            "id": 0,
	            "name": "Humanface"
	        },
	        {
	            "id": 1,
	            "name": "Humanhand"
	        }
	    ],
	    "date": "2018-11-22T12:32:26.145586",
	    "results": [
	        {
	            "file_path": "{inference_image_path}",
	            "prediction": [
	                {
	                    "box": [
	                        1306.8501420021057,
	                        409.00918841362,
	                        587.2840890884399,
	                        939.6917223930359
	                    ],
	                    "class": {
	                        "id": 0,
	                        "name": "Humanface"
	                    },
	                    "score": "0.6719009876251221"
	                }
	            ]
	        }
	    ],
	    "task": "IMAGE.OBJECT_DETECTION",
	    "version": 0.2
	}
	```