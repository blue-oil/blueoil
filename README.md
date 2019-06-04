<div align="center">
  <img src="https://s3-ap-northeast-1.amazonaws.com/leapmind-public-storage/img/blueoil_cover.png">
</div>

---

[![Build Status](https://jenkins.blue-oil.org/job/blueoil_main/badge/icon)](https://jenkins.blue-oil.org/job/blueoil_main/)
[![Build Status](https://jenkins.blue-oil.org/job/blueoil_lmnet/badge/icon)](https://jenkins.blue-oil.org/job/blueoil_lmnet/)
[![CircleCI](https://circleci.com/gh/blue-oil/blueoil.svg?style=svg)](https://circleci.com/gh/blue-oil/blueoil)

Blueoil provides two features.
* Training a neural network model
* Converting a trained model to an executable binary (or library), which utilize FPGAs for acceleration.

## Documentation

You can see **[online documentation](https://docs.blue-oil.org)** with enter.

Check out the [Installation](https://docs.blue-oil.org/install/install.html) and [Usage Guide](https://docs.blue-oil.org/usage/index.html) page for getting started.


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
$ ./blueoil_test.sh

Usage: ./blueoil_test.sh <YML_CONFIG_FILE(optional)>

Arguments:
  YML_CONFIG_FILE       config file path for this test (optional)
```

### Tests for CI
* [Jenkins settings for CI](./tests/README.md)
