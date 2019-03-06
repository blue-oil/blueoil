# Training for Basic Classification

This guide trains a neural network model to classify images of 10 objects in the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset on GPU server.

<img src="../_static/cifar10.png" width="400">

## Preparation

The CIFAR-10 dataset is available from official website, but original dataset's format (numpy array) is unsupported in the Blueoil.

Blueoil supports 2 formats for basic classification.

- Caltech101 format
- DeLTA-Mark format

Note: *Please see the detail in <a href="../usage/dataset.html">Prepare training dataset</a>*


You can download other data format (PNG image format) of CIFAR-10 from our mirror site.
[https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/cifar.tgz](https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/cifar.tgz)

    $ wget https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/datasets/cifar.tgz
    $ tar xzf cifar.tgz

The subdirectory of this dataset under train or test becomes the class name and images are located under the subdirectory. (This is same as Caltech101 format)
```
cifar
 ├─ train
 │   ├─ airplane
 │   │    ├─ xxxx_airplane.png
 │   │    ├─ yyyy_airplane.png
 │   │    ...
 │   ├─ automobile
 │   │    ├─ ...
 │   ├─ bird
 │   │    ├─ ...
 │   ├─ ...
 │   └─ truck
 │        ├─ ...
 └─ test
     ├─ airplane
     │    ├─ ...
     ├─ ...
     ...
```

CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Generate a configuration file

Generate your model configuration file interactively by running `blueoil init` command.

    $ ./blueoil.sh init

This is an example of configuration.

```
#### Generate config ####
  your model name ():  cifar10_test
  choose task type:  classification
  choose network:  LmnetV1Quantize
  choose dataset format:  Caltech101
  training dataset path:  {dataset_dir}/cifar/train/
  set validataion dataset? (if answer no, the dataset will be separated for training and validation by 9:1 ratio.)  yes
  validataion dataset path:  {dataset_dir}/cifar/test/
  batch size (integer):  64
  image size (integer x integer):  32x32
  how many epochs do you run training (integer):  100
```

- Model name: (Any)
- Task type: classification
- Network: LmnetV1Quantize / LmnetV0Quantize
- Dataset format: Caltech101
- Dataset path:
    - Training: {dataset_dir}/cifar/train/
    - Validation: {dataset_dir}/cifar/test/
- Batch size: (Any)
- Image size: 32x32
- Number of epoch: (Any number)

If configuration finishes, configuration file is generated `{Model name}.yml` under `./config` directory.

## Train a neural network

Train your model by running `blueoil train` command with model configuration.

    $ ./blueoil.sh train config/{Model name}.yml

When training is started, training log and checkpoints are generated under `./saved/{Mode name}_{TIMESTAMP}` directory.

Training is running on TensorFlow backend. So you can use TensorBoard to visualize your training process.

    $ ./blueoil.sh tensorboard saved/{Model name}_{TIMESTAMP} {Port}

- Loss / Cross Entropy, Loss, Weight Decay
<img src="../_static/train_loss.png">


- Metrics / Accuracy
<img src="../_static/train_metrics.png">

## Convert training result to FPGA ready format.

Convert trained model to executable binary files for x86, ARM, and FPGA.
Currently, conversion for FPGA only supports Intel Cyclone® V SoC FPGA.

    $ ./blueoil.sh convert config/[Model name].yml saved/{Mode name}_{TIMESTAMP}

`Blueoil convert` automatically executes some conversion processes.
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

- Prepare inference images (not included in the training dataset)

    [ for example: airplane ]
    <img src="../_static/airplane.png" width=128>


- Run inference script

    Explore into the `output/python` directory, and
    run `run.py` and inference result is saved in `./output/output.json`.

    Note: If you run the script for the first time, you have to setup a python environment (2.7 or 3.5+) and install requirements python packages.

      $ cd {output/python directory}
      $ sudo pip install -r requirements.txt  # only the first time
      $ python run.py \
          -i {inference image path} \
          -m ../models/lib/lib_x86.so \
          -c ../models/meta.yaml

- Check inference result

      {
          "classes": [
              { "id": 0, "name": "airplane" },
              { "id": 1, "name": "automobile" },
              ...
              { "id": 9, "name": "truck" }
          ],
          "date": "yyyy-mm-ddThh:mm:ss.nnnnnnn",
          "results": [
              {
                  "file_path": "{inference image path}",
                  "prediction": [
                      {
                          "class": { "id": 0, "name": "airplane" },
                          "probability": "0.95808434"
                      },
                      {
                          "class": { "id": 1, "name": "automobile" },
                          "probability": "0.0007377896"
                      },
                      ...
                      {
                          "class": { "id": 9, "name": "truck" },
                          "probability": "0.008015169"
                      }
                  ]
              }
          ],
          "task": "IMAGE.CLASSIFICATION",
          "version": 0.2
      }
