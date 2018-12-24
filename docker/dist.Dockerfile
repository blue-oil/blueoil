ARG base_docker_image="blueoil/blueoil:master"

FROM ${base_docker_image}

MAINTAINER masuda@leapmind.io

WORKDIR /home/blueoil

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

# Set env to install horovod with nccl and tensorflow option
ENV HOROVOD_GPU_ALLREDUCE NCCL
ENV HOROVOD_WITH_TENSORFLOW 1
# Set temporarily CUDA stubs to install Horovod
RUN ldconfig /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs
# Install requirements for distributed training
RUN pip install -r lmnet/dist.requirements.txt
# Unset temporarily CUDA stubs
RUN ldconfig

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0 --mca btl_vader_single_copy_mechanism none
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
