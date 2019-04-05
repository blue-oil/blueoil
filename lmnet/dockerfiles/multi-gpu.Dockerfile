FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER wakisaka@leapmind.io

# TensorBoard
EXPOSE 6006

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

RUN echo "deb http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse \n\
deb-src http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Pillow and matplotlib has many dependencies for display.
RUN apt-get update && apt-get install -y \
    python3-pil \
    libjpeg8-dev \
    zlib1g-dev \
    python3-matplotlib \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install NCCL
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV NCCL_VERSION=2.1.15-1+cuda8.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnccl2=$NCCL_VERSION \
    libnccl-dev=$NCCL_VERSION

# Install requirements for OpenMPI and Horovod
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev

# Install OpenMPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
    tar zxf openmpi-3.0.0.tar.gz && \
    cd openmpi-3.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun && \
    chmod a+x /usr/local/bin/mpirun

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

RUN pip install -U pip setuptools

COPY requirements.txt /tmp/requirements.txt
COPY dev.requirements.txt /tmp/dev.requirements.txt
COPY test.requirements.txt /tmp/test.requirements.txt
COPY docs.requirements.txt /tmp/docs.requirements.txt
COPY gpu.requirements.txt /tmp/gpu.requirements.txt
COPY dist.requirements.txt /tmp/dist.requirements.txt

WORKDIR /home/lmnet

# Install requirements
RUN pip install -r /tmp/gpu.requirements.txt

# Set env to install horovod with nccl and tensorflow option
ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_WITH_TENSORFLOW 1
# Set temporarily CUDA stubs to install Horovod
RUN ldconfig /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs
# Install requirements for distributed training
RUN pip install -r /tmp/dist.requirements.txt
# Unset temporarily CUDA stubs
RUN ldconfig

# Build coco. It needs numpy.
COPY third_party third_party
# https://github.com/cocodataset/cocoapi/blob/440d145a30b410a2a6032827c568cff5dc1d2abf/PythonAPI/setup.py#L2
RUN cd third_party/coco/PythonAPI && pip install -e .

# For development 
RUN apt-get update && apt-get install -y \
    x11-apps \
    imagemagick \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0 --mca btl_vader_single_copy_mechanism none
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
