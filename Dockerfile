ARG UBUNTU_VERSION=18.04
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu${UBUNTU_VERSION}
# python3.6 is provided by ubuntu18.04
# python3.5 is provided by ubuntu16.04

MAINTAINER blueoil-admin@leapmind.io

# TensorBoard
EXPOSE 6006

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
ENV DEBIAN_FRONTEND noninteractive

# Install basic packages
RUN apt-get update && apt-get install -y \
    cmake \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install dependencies for Pillow, Scipy and matplotlib for display
RUN apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libjpeg8-dev \
    liblapack-dev \
    llvm \
    make \
    python3-matplotlib \
    python3-pil \
    wget

# install aarch64 cross compile environment
RUN apt-get install -y --no-install-recommends crossbuild-essential-arm64

# Install x-compiler
RUN apt-get install -y g++-5-arm-linux-gnueabihf && \
    ln -s /usr/bin/arm-linux-gnueabihf-g++-5 /usr/bin/arm-linux-gnueabihf-g++

# Install requirements
RUN pip3 install -U pip setuptools
COPY lmnet/*requirements.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/gpu.requirements.txt && rm -rf /tmp/requirements

# In order to install blueoil requirements `prompt_toolkit==1.0.15`, uninstall prompt-toolkit v2.0 that depends on `pdb==0.10.2`.
RUN pip uninstall -y prompt-toolkit

# Build coco. It needs numpy.
COPY lmnet/third_party /home/blueoil/lmnet/third_party
# https://github.com/cocodataset/cocoapi/blob/440d145a30b410a2a6032827c568cff5dc1d2abf/PythonAPI/setup.py#L2
RUN cd /home/blueoil/lmnet/third_party/coco/PythonAPI && pip install -e .

# For development
RUN apt-get install -y --no-install-recommends \
    x11-apps \
    imagemagick \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy blueoil sources to docker image
COPY blueoil /home/blueoil/blueoil
COPY setup.* /home/blueoil/
COPY output_template /home/blueoil/output_template
# Install blueoil
WORKDIR /home/blueoil
RUN python3 setup.py install
RUN chmod 777 /home/blueoil

# Copy dlk sources to docker image
COPY dlk /home/blueoil/dlk
# Install dlk
WORKDIR /home/blueoil/dlk
RUN PYTHONPATH=python/dlk python3 setup.py install
RUN chmod 777 /home/blueoil/dlk

# Copy lmnet sources to docker image
COPY lmnet /home/blueoil/lmnet

ENV PYTHONPATH $PYTHONPATH:/home/blueoil:/home/blueoil/lmnet:/home/blueoil/dlk/python/dlk
WORKDIR /home/blueoil
