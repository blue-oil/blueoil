# lmnet in Docker

To streamline the installation process, we have Dockerfile and docker-compose.yml, so you can get started with lmnet in few minutes.

## Pre requisites

- Docker >= 17.12.0
- docker-compose >= 1.21.0
- nvidia-docker >= 2.0

## Environment variable
You can set following variables before running training commands.

- `OUTPUT_DIR` : Directory for outputs, the default is `"./saved"`
- `DATASET_DIR` : Directory for datasets, the default is `"/storage/dataset"`
- `CUDA_VISIBLE_DEVICES` : GPU's device number, the default is `"0"`

You can set by your own variables by using `export`, ex:`export OUTPUT_DIR="/storage/lmnet/saved"`

## Running on a single machine
### CPU mode
#### Building container for using CPU only
```
$ docker-compose build cpu-tensorflow
```
#### Running container for training with CPU only
```
$ docker-compose run --rm cpu-tensorflow python executor/train.py [some options]
```

### single GPU mode
#### Building container for using single GPU
```
$ docker-compose build tensorflow
```
#### Running container for training with single GPU
```
$ docker-compose run --rm tensorflow python executor/train.py [some options]
```
You can also select the GPU which you want to use by setting `CUDA_VISIBLE_DEVICES`
You can see device number by `nvidia-smi` command.
This is an example for using gpu:2.
```
$ export CUDA_VISIBLE_DEVICES="2"
```

### multi GPU mode
#### Building container for using multi GPUs
```
$ docker-compose build horovod
```
#### Running container for distributed training with multi GPUs
You can run distributed training by using `mpirun` command.
This distribution is based on Uber's open source distributed training framework: [Horovod](https://github.com/uber/horovod)
You should set option `mpirun -np [number of processes]` (`[number of processes]` is how many workers you distribute).
Each process uses one GPU for training, so `[number of processes]` should be less than or equal to the number of GPUs on your machine. And you should set `CUDA_VISIBLE_DEVICES` to use multi GPUs because it's default is "0" (only one GPU).
This is an example of cifar10 with 4GPUs.
```
$ export CUDA_VISIBLE_DEVICES="0,1,2,3"
$ docker-compose run --rm horovod mpirun -np 4 python executor/train.py -c configs/core/classification/lmnet_quantize_cifar10_distribute.py -i test_for_distribution
```
There are sample configs for distributed training.
```
configs/core/classification/lmnet_quantize_cifar10_distribute.py
configs/core/classification/lmnet_quantize_cifar100_distribute.py
```
These configs have the parameter of `NUM_WORKER`, and are designed to scale learning rate with `NUM_WORKER`.
You should set `NUM_WORKER` as same as `mpirun`'s option of `[number of processes]`.
The effective batch size in synchronous distributed training is scaled by the number of workers. So it is need to increase in learning rate to compensates for the increased batch size.

## Running on the DGX-1
The DGX-1 has 8 GPUs(Tesla V100), and V100(volta architecture) needs CUDA 9 or higher.
Current lmnet requires tensorflow 1.4, and it is not supported CUDA 9, so we should use docker images provided by NVIDIA and optimoized to run tensorflow 1.4 with V100.
We can run lmnet with NVIDIA's docker image by using docker-compose service `horovod-volta`.

### Getting started
#### Login to NGC (NVIDIA GPU Cloud)
You should login to NGC to pull NVIDIA's official docker images
```
$ API_KEY=[Your NGC API KEY]
$ docker login nvcr.io -u \$oauthtoken -p ${API_KEY}
```

#### Building container for volta
```
$ docker-compose build horovod-volta
```

#### Running container for training with one GPU
You should set `CUDA_VISIBLE_DEVICES` to use only selected one GPU.
```
$ export CUDA_VISIBLE_DEVICES="0"
$ docker-compose run --rm horovod-volta python executor/train.py [some options]
```

#### Running container for distributed training with multi GPUs
The default of `CUDA_VISIBLE_DEVICES` is set as `1,0,2,3,7,6,4,5`.
This reason is bandwidth optimization with horovod's ring-allreduce algorithm on the architecture of NVLINK in DGX-1.
Before training, you should set `NUM_WORKER` in your config as same as `mpirun`'s option of `[number of processes]`.
This is an example of cifar10 with 8GPUs.
```
$ docker-compose run --rm horovod-volta mpirun -np 8 python executor/train.py -c configs/core/classification/lmnet_quantize_cifar10_distribute.py -i test_for_distribution
```

## Running Tensorboard
You can run tensorboard as following command.
```
$ docker-compose run --rm tensorboard
```
The default port is `6006` and you can see all results under your `OUTPUT_DIR` as default.
You can change port as following,
```
$ docker-compose run --rm tensorboard tensorboard /storage/lmnet/saved --port [your port]
```
`/storage/lmnet/saved` is docker's output dir linked with your host's OUTPUT_DIR, so you need not change it.
If you want to see only one result of your experiment, you can run as follow,
```
$ docker-compose run --rm tensorboard tensorboard /storage/lmnet/saved/[experiment_id] --port [your port]
```
`experiment_id` is same as set by training command's -i option.


# Cross build for benchmark on ARM of Cyclone V Soc.
In order to measure latency benchmark on ARM of Cyclone V Soc, do bellow.

On X86
* Build ARM docker image on X86 with `qemu` to install dependency python packages.
* Export builded docker container as tar file.

On Cyclone V
* Use `chroot` to enter exported docker container environment.
* Execute lmnet script.

The docker image are installed CycloneV tensorflow builded by [tensorflow-on-arm](https://github.com/lhelontra/tensorflow-on-arm) with the [cyclone_v.conf](https://github.com/LeapMind/lmnet/blob/master/third_party/override/tensorflow-on-arm/build_tensorflow/configs/cyclone_v.conf), also installed `lmnet` dependency libraries.


## Required
```
sudo apt-get install binfmt-support qemu-user-static
```

## Cross build and export the ARM docker image on X86.
build docker iamge.
```
docker-compose build cyclone-v-cross
```

docker container export to tar file.
```
docker export `docker run cyclone-v-cross && docker ps -a -q | head -n 1` > cyclone-v-cross.tar

```

## chroot, execute script for benchmark on Cyclone V board.

unpack the tar file.
```
$ mkdir lmnet_root
$ tar xf cyclone-v-cross.tar -C lmnet_root/
```

chroot, execute script.
```
$ chroot lmnet_root

# in changed root.
$ cd /home/lmnet
$ LC_ALL=en_US.UTF-8 PYTHONPATH=. python executor/measure_latency.py -c configs/example/classification.py
```
