<div align="center">
  <img src="https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/img/blueoil_cover.png">
</div>

---

Blueoil provides two features.
* Training a neural network model
* Converting a trained model to an executable binary (or library), which utilize FPGAs for acceleration.

| Type | Status |
| --- | --- |
| blueoil | [![Build status](https://badge.buildkite.com/c56e1c6e8160a5351fc2aa19dce80705b1aa8426ad322cf9e3.svg?branch=master)](https://buildkite.com/blueoil/blueoil-test) |
| lmnet | [![Build status](https://badge.buildkite.com/45ff7e206fc1de4c160f72781463fdbbcffb1321c1e69e08d1.svg?branch=master)](https://buildkite.com/blueoil/lmnet-test) |
| dlk | [![Build status](https://badge.buildkite.com/c1d2082e8076b48057a621c7dbabfa280975dcd71da83f49e9.svg?branch=master)](https://buildkite.com/blueoil/dlk-test) |
| docs | [![CircleCI](https://circleci.com/gh/blue-oil/blueoil.svg?style=svg)](https://circleci.com/gh/blue-oil/blueoil) |

See also [CI settings](./tests/README.md).
## Documentation

You can see **[online documentation](https://docs.blueoil.org)** with enter.

Check out the [Installation](https://docs.blueoil.org/install/install.html) and [Usage Guide](https://docs.blueoil.org/usage/index.html) page for getting started.


**Note**: Currently, Installation page is just in to be written, Please see [Setup](#set-up) section to build docker on your development environment.


## Prerequisites
- GNU/Linux x86_64 with kernel version > 3.10
- NVIDIA GPU with Architecture >= 3.0 (Kepler)
- NVIDIA drivers >= 410.48
- Docker >=1.12 (or >=17.03.0)
- nvidia-docker >= 2.0

The blueoil is run on docker container with original docker image based on NVIDIA's [CUDA images](https://github.com/NVIDIA/nvidia-docker/wiki/CUDA#requirements) (cuda:10.0-cudnn7-devel).

The machine running the CUDA container only requires the NVIDIA driver, the CUDA toolkit doesn't have to be installed.

Please see the detail in the nvidia-docker's [prerequisites](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)#prerequisites).

## Set up
There are some submodules in this repositry, so you should run `git submodule update --init --recursive` after cloning or `git clone --recursive [this repository]`.
```
make build
```
Note: The private repository submodules are set to connect by ssh, if you want to use HTTPS, you should edit URLs in `.gitmodules` and run `git submodule sync` before `git submodule update --init --recursive` command. ([see how to edit](https://stackoverflow.com/a/30885128))


## How to make document

```
cd docs
make html
```

You can see generated documents in HTML format under `docs/_build/html/` directory on your enviroment.

Also, you can see the deploy-preview online documentation from a `Pull Request` page that are integrated by [netilify](http://netlify.com).


## How to test
We can test each opereations of drore_run.sh by using shell script.

### Prerequisites
- `expect` >= version 5.45

```
$ make test
```

You can test specific task.

```
$ CUDA_VISIBLE_DEVICES={YOUR_GPU_ID} make test-classification
$ CUDA_VISIBLE_DEVICES={YOUR_GPU_ID} make test-object-detection
$ CUDA_VISIBLE_DEVICES={YOUR_GPU_ID} make test-semantic-segmentation
```

You can also test the modules used in Blueoil.

```
$ make test-lmnet
$ make test-dlk
```
