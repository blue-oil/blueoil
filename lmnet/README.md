# LMNet
This project is developed as one part of Blueoil project -- the deep learning model part.  


* Training and Evaluation
  in `executor` dir
  * `train.py`: entry point script for training.
  * `evaluate.py`: entry point script for evaluattion metrics of trained models.
  * `output_event.py`: entry point script for output metrics in csv and markdown from tensorboard event log.
  
* Blueoil integration
  in `executor` dir.
  * `export.py`: entry point script for exporting proto buffer file from a trained model for DLK converter.
  * `predict.py`: entry point script for inference of trained model, loading none-labeled images and outputs result json and npy.

* Utils
  * `measure_latency.py`: entry point script for measuring inference latency.
  * `convert_weight_from_darknet.py`: entry point script for convert weight format form darknet framework.

- - -

# Getting Started
Here is a simple example on how to use this tool.  
You can see more usage details in the following sub-sections. (e.g. downloading dataset, training)
```
# clone this repository
git clone --recursive git@github.com:LeapMind/blueoil.git
cd lmnet

# make 'dataset' directory to store dataset.
mkdir -p dataset/CIFAR_10

# download Cifar10 dataset.
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
mv cifar-10-python.tar.gz dataset/CIFAR_10/

# decompress
cd dataset/CIFAR_10 && tar xvf cifar-10-python.tar.gz && cd ../..

# require python version more than `3.5`. create and move to virtual environment.
python -m venv .venv
source .venv/bin/activate

# We need to install cython and numpy before "pip install -r" for pycocotools
pip install --upgrade cython numpy

# install python requirements
pip install -r ../cpu.requirements.txt

# start sample training by lmnet_v0 and cifar-10. It takes few minutes.
PYTHONPATH=. python executor/train.py -c configs/example/classification.py

# after training or on another terminal, serve tensorboard. you can see tensorboard in http://localhost:6006/
tensorboard --logdir saved/experiment/
```
If you want to use Docker, read the [lmnet in Docker](docs/docker/README.md) page.
- - -

# Setting up python environment
Before you start to work on this project, you need to setup your local python environment correctly.  

## Supported python version
The following versions of Python can be used: `3.5`, `3.6`.

## Installing python packages
To manage the version, we are using `requirements.txt`
style.
- only CPU
`pip install -r ../cpu.requirements.txt`
- with GPU
`pip install -r ../gpu.requirements.txt`


- - -

# Data Conventions for pre/post-process
* Image
  * The order of color channel is `RGB`.
  * The Size is represented by `(height, width)`.
  * The range of pre-process and data augmentation's inputs is `[0, 255]`.
    The network input's range depends on pre-process and augmentation.

* Bounding Boxes
  * The Coordinate's order is `(x(left), y(top), width, height)`.
  * A Ground truth box is `(x(left), y(top), width, height, class_id)`.
    Considering the batch size, Dataset class feeds `[batch size, number of max boxes, 5(x, y, w, h, class_id)]` shape data.
  * A predicted box is `(x(left), y(top), width, height, class_id, score)`.
    Considering the batch size, the predicted boxes shape is `[batch size, number of predict boxes, 6(x, y, w, h, class_id, score)]`
    For `YOLOv2` network, The post process `FormatYOLOV2` convert bare convolution network output to the shape.
    Then, all othres post process for object detection (ie. `NMS`) assume the shape.

- - -

# Training
In this project, we can choose model architecture and dataset separately. (VGG->MNIST, VGG->Fruits etc.)
1. Network
    e.g. VGG16, Lmnet(original model)
1. Dataset
    e.g. MNIST, Fruits(original dataset)

## Downloading datasets
to be written.

## Starting training
Main training script is `executor/train.py`.
When you start training, you need to specify some (required) options.  
You can see the option descriptions with `-h` flag.
```
# PYTHONPATH=. python executor/train.py -h 
Usage: train.py [OPTIONS]

Options:
  -c, --config_file TEXT    config file path for this training  [required]
  -i, --experiment_id TEXT  id of this training  [required]
  --recreate                delete and recreate experiment id dir
  -n, --network TEXT        network name which you want to use for this
                            training. override config.DATASET_CLASS
  -d, --dataset TEXT        dataset name which is the source of this training.
                            override config.NETWORK_CLASS
  -h, --help                Show this message and exit.
```

`--network` and `--dataset` option will override config on-the-fly.  
If you'd like to use your own custom config, please refer to `configs/example/classification.py` before your training. See also [Config specification](../docs/specification/config.md).
To run training in Docker, read the [lmnet in Docker](docs/docker/README.md) page.

## Saving model and training results
After training a model, you can save the result in the filesystem.  
By default, these files are generated under `saved` directory.  
You can change the directory via unix env variable `OUTPUT_DIR`, please refer to `lmnet/environment.py`.

Under the `saved` directory, the experiments are saved with an associated id.  
This `id` is specified via the script option `-i` or `--experiment_id`.  
e.g.
`PYTHONPATH=. python executor/train.py -n classification.lmnet -d mnist -i lmnet_mnist`  
In the above case, id and experiment directory name is `lmnet_mnist`.  
Default value of `--experiment_id` is `experiment`.  

And then, sub-directories are also created in `lmnet_mnist`.  
Currently, there are 3 sub-directories.

- config.py
Actual copy of the config file used for training.  
It is useful to record the configured parameters, lest we forget it.
- checkpoints  
tensorflow's ckpt files are saved there
- tensorboard  
tensorboard's source files are saved there

Finally, The tree structure is per below.
```
saved/
  | - lmnet_mnist  # your own directory name
          | - config.py
          | - tensorboard
                   | some files...
          | - checkpoints
                   | some files...
  | - experiment  # default directory name
          | - config.py
          | - tensorboard
                   | some files...
          | - checkpoints
                   | some files...
```


## Starting evaluation
Main evaluation script is `executor/evaluate.py`.  
When you start doing evaluation, you need to specify some (required) options.  
You can see the option descriptions with `-h` flag.
```
# PYTHONPATH=. python executor/evaluate.py -h 
Usage: evaluate.py [OPTIONS]

Options:
  -i, --experiment_id TEXT  id of this training  [required]
  --restore_path TEXT       restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001
  -n, --network TEXT        network name. override config.NETWORK_CLASS
  -d, --dataset TEXT        dataset name. override config.DATASET_CLASS
  -c, --config_file TEXT    config file path. override(merge) saved experiment config.
                            if it is not provided, it restore from saved experiment config.
  -o, --output_dir TEXT     Output directory to save a evaluated result
  -h, --help                Show this message and exit.
```

`--network` and `--dataset` option override config on the fly.  

- - -

## Improving accuracy
In progress.  
We need to implement some ideas to increase the accuracy.

### Using train validation saving
A subset is split from the train data and used to decide points at which the model is saved.  
This can increase performance of the saved model for quantized network, but also increases training time.  

The SAVE_CHECKPOINT_STEPS parameter should be small to make use of this feature (increasing the value of this parameter gives faster training but might not increase accuracy as much).  
Also TRAIN_VALIDATION_SAVING_SIZE determines how much to split from the train data.  
Currently, this feature is implemented for quantized classification of cifar10 and cifar100.
If you don't want to use this feature, set TRAIN_VALIDATION_SAVING_SIZE to zero.  

The KEEP_CHECKPOINT_MAX is equivalent to 'max_to_keep' of tensorflow train.Saver parameter which indicates the maximum number of recent checkpoint files to keep. As new files are created, older files are deleted. 
If None or 0, no checkpoints are deleted from the filesystem but only the last one is kept in the checkpoint file. Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)

To apply this feature to another dataset, the dataset file should define another available subset called train_validation_saving, which is split from the original train dataset in the dataset file. Also a dataset parameter TRAIN_VALIDATION_SAVING_SIZE should be included in the config file.

- - -

# Exporting model to proto buffer
Exporting a trained model to proto buffer files and meta config yaml.

In the case with `images` option, create each layer output value npy files in `export/{restore_path}/{image_size}/{image_name}/**.npy` for debug.

* Load config file from saved experiment dir.
* Export config file to yaml. See also [Config specification](docs/specification/config.md).
  * `config.yaml` can be used for training and evaluation in python. i.e. [classification.yaml](configs/example/classification.yaml) is exported from [classification.py](configs/example/classification.py)
  * `meta.yaml` include only few parameter for application such as demo. i.e. [classification_meta.yaml](configs/example/classification_meta.yaml) is exported from [classification.py](configs/example/classification.py)
* Save the model protocol buffer files (tf) for DLK converter.
* Output each layer npy files for DLK converter debug.
* Write summary in tensorboard `export` dir.

```
# PYTHONPATH=. python executor/export.py -h
Usage: export.py [OPTIONS]

  Exporting a trained model to proto buffer files and meta config yaml.

  In the case with `images` option, create each layer output value npy files
  in `export/{restore_path}/{image_size}/{image_name}/**.npy` for debug.

Options:
  -i, --experiment_id TEXT        id of this experiment.  [required]
  --restore_path TEXT             restore ckpt file base path. e.g.
                                  saved/experiment/checkpoints/save.ckpt-10001
  --image_size <INTEGER INTEGER>...
                                  input image size height and width. if it is
                                  not provided, it restore from saved
                                  experiment config. e.g. --image_size 320 320
  --images TEXT                   path of target images
  -c, --config_file TEXT          config file path. override saved experiment
                                  config.
  -h, --help                      Show this message and exit.
```

e.g.
`PYTHONPATH=. python executor/export.py -i lmnet_cifar10 --restore_path saved/lmnet_cifar10/checkpoints/save.ckpt-99001 --images apple_128.png --images apple.png --image_size 128 192`


# Measuring latency
Measure the average latency of certain model's prediction at runtime.

The latency is averaged over number of repeated executions -- by default is to run it 100 times.
Each execution is measured after tensorflow is already initialized and both model and images are loaded.
Batch size is always 1.

Measure two types latency,
First is `overall` (including pre-post-processing which is being executed on CPU), Second is `network-only` (model inference, excluding pre-post-processing).

```
Options:
  -c, --config_file TEXT          config file path.
                                  When experiment_id is
                                  provided, The config override saved
                                  experiment config. When experiment_id is
                                  provided and the config is not provided,
                                  restore from saved experiment config.
  -i, --experiment_id TEXT        id of this experiment.
  --restore_path TEXT             restore ckpt file base path. e.g. saved/expe
                                  riment/checkpoints/save.ckpt-10001.
  --image_size <INTEGER INTEGER>...
                                  input image size height and width. if it is
                                  not provided, it restore from saved
                                  experiment config.
  -n, --step_size INTEGER         number of execution (number of samples).
                                  default is 100.
  --cpu                           flag use only cpu
  -h, --help                      Show this message and exit.
```

e.g.
```
$ PYTHONPATH=. python executor/measure_latency.py -c configs/core/object_detection/lm_fyolo_quantize_pascalvoc_2007_2012.py -n 20 --image_size 160 160

...
---- measure latency result ----
total number of execution (number of samples): 20
network: LMFYoloQuantize
use gpu by network: True
image size: [160, 160]
devices: ['device: 0, name: GeForce GTX TITAN X, pci bus id: 0000:01:00.0, compute capability: 5.2']

* overall (include pre-post-process which execute on cpu)
total time: 101.7051 msec
latency
   mean (SD=standard deviation): 5.0853 (SD=0.1654) msec, min: 4.9169 msec, max: 5.5995 msec
FPS
   mean (SD=standard deviation): 196.8454 (SD=6.1032), min: 178.5874, max: 203.3799

* network only (exclude pre-post-process):
total time: 78.3966 msec
latency
   mean (SD=standard deviation): 3.9198 (SD=0.1583) msec, min: 3.7389 msec, max: 4.4239 msec
FPS
   mean (SD=standard deviation): 255.5047 (SD=9.7093), min: 226.0471, max: 267.4598
---- measure latency result ----
```



# Making prediction
Make predictions from input dir images by using trained model.

Save the predictions npy, json, images results to output dir.
* npy: `{output_dir}/npy/{batch number}.npy`
* json: `{output_dir}/json/{batch number}.json`
* images: `{output_dir}/images/{some type}/{input image file name}`

The output predictions Tensor(npy) and json format depends on task type. Plsease see [Output Data Specification](docs/specification/output_data.md).

```
python3 executor/predict.py -h
WARNING: The PYTHONPATH variable is not set. Defaulting to a blank string.
Usage: predict.py [OPTIONS]

Options:
  -in, --input_dir TEXT           Input directory which contains images to
                                  make predictions  [required]
  -o, --output_dir TEXT           Output directory to save a predicted result
                                  [required]
  -i, --experiment_id TEXT        Experiment id  [required]
  -c, --config_file TEXT          config file path. override saved experiment
                                  config.
  --restore_path TEXT             restore ckpt file base path. e.g.
                                  saved/experiment/checkpoints/save.ckpt-10001
  --save_images / --no_save_images
                                  Flag of saving images. Default is True.
  -h, --help                      Show this message and exit.
```

e.g.
`PYTHONPATH=. python executor/predict.py -in ./dataset/images -o ./outputs -i lmnet_cifar10`


# Convert weight from darknet
Weight converter form darknet framework to tensorflow checkpoints file.
You can convert [Yolov2](https://pjreddie.com/darknet/yolov2/) and [Darknet19](https://pjreddie.com/darknet/imagenet/#darknet19_448) network weights.

Please download darknet weights at `inputs` dir.
```
cd inputs
wget http://pjreddie.com/media/files/darknet19_448.weights
wget https://pjreddie.com/media/files/yolo-voc.weights
```

After execute `convert_weight_from_darknet.py`, You can get checkpoints file on
* darknet19: `${OUTPUT_DIR}/convert_weight_from_darknet/darknet19/checkpoints/save.ckpt`
* yolov2: `${OUTPUT_DIR}/convert_weight_from_darknet/yolo_v2/checkpoints/save.ckpt`

```
# PYTHONPATH=. python executor/convert_weight_from_darknet.py -h
Usage: convert_weight_from_darknet.py [OPTIONS]

Options:
  -m, --model [yolov2|darknet19]  yolo2 or darknet19  [required]
  -h, --help                      Show this message and exit.
```

e.g.
`PYTHONPATH=. python executor/convert_weight_from_darknet.py -m yolov2`


# Profiling model
Profiling a trained model.

If it exists unquantized layers, use `-uql` to point it out.

```
# PYTHONPATH=. python executor/profile_model.py -h
Usage: profile_model.py [OPTIONS]

  Profiling a trained model.

  If it exists unquantized layers, use `-uql` to point it out.

Options:
  -i, --experiment_id TEXT     id of this experiment.
  --restore_path TEXT          restore ckpt file base path. e.g.
                               saved/experiment/checkpoints/save.ckpt-10001
  -c, --config_file TEXT       config file path.
                               When experiment_id is
                               provided, The config override saved experiment
                               config. When experiment_id is provided and the
                               config is not provided, restore from saved
                               experiment config.
  -b, --bit INTEGER            quantized bit
  -uql, --unquant_layers TEXT  unquantized layers
  -h, --help                   Show this message and exit.
```

e.g.
`PYTHONPATH=. python executor/profile_model.py -i lmnet_cifar10 --restore_path saved/lmnet_cifar10/checkpoints/save.ckpt-99001 --bit 2 -uql conv1 -uql conv7`

`PYTHONPATH=. python executor/profile_model.py -c configs/core/classification/lmnet_cifar10.py --bit 1`

- - -

# Test code
- Before your branch can be merged into `master` branch, your code has to pass all tests.
- When you create a new `pull requests`, please **make sure** your code has passed the following tests in your local environment.

## How to test locally:
Go to project root and run following commands:
- all: `tox`
- flake8: `tox -e flake8`
- pytest: `tox -e pytest`

If your code was running in `docker`, go to project root and run:
- all: `docker-compose run --rm tensorflow tox -e py36`
- flake8: `docker-compose run --rm tensorflow tox -e flake8`
- pytest: `docker-compose run --rm tensorflow tox -e pytest`


# Docs

For convenience, just run `remake_docs.sh` under `docs`.
```
./remake_docs.sh port
```
For example, `./remake_docs.sh 8000`
will update `source`. Then build html and serve at port `8000`.

For more details, please take a look at the script file. 

Or you can just run step by step:

Build
```
$ cd docs
# Add lmnet/lmnet dir to PYTHONPATH for better-apidoc bug.
# https://github.com/goerz/better-apidoc/issues/9
$ export PYTHONPATH=../lmnet:$PYTHONPATH
$ better-apidoc -t source/_templates/ -feo ./source ../lmnet/
```

Make html. Run liveload web server
```
$ cd docs
$ sphinx-autobuild source _build/html/
```
