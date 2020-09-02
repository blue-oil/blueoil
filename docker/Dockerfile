FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 AS base

LABEL maintainer="admin@blueoil.org"

# TensorBoard
EXPOSE 6006

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda-10.0 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    fonts-dejavu \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-pil \
    libjpeg8-dev \
    libpng-dev \
    zlib1g-dev \
    liblapack-dev \
    git \
    make \
    build-essential \
    wget \
    g++-8 \
    g++-8-aarch64-linux-gnu \
    g++-8-arm-linux-gnueabihf \
    openssh-client \
    openssh-server \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 80 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
RUN ln -s /usr/bin/arm-linux-gnueabihf-g++-8 /usr/bin/arm-linux-gnueabihf-g++
RUN ln -s /usr/bin/aarch64-linux-gnu-g++-8 /usr/bin/aarch64-linux-gnu-g++

# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /var/run/sshd

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install OpenMPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.1.tar.gz && \
    tar zxf openmpi-4.0.1.tar.gz && \
    cd openmpi-4.0.1 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Configure OpenMPI to run good defaults:
#   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0 --mca btl_vader_single_copy_mechanism none
RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf

WORKDIR /home/blueoil
RUN pip3 install -U pip setuptools

FROM base AS blueoil-base

# COPY resources required for blueoil installation
COPY setup.* /home/blueoil/
COPY blueoil /home/blueoil/blueoil
COPY output_template /home/blueoil/output_template

# Set env to install horovod with nccl and tensorflow option
ENV HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_WITH_TENSORFLOW=1

# Install requirements for distributed training temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    pip install -e .[gpu,test,docs] && \
    pip install -e .[dist] && \
    ldconfig
RUN pip install pycocotools==2.0.0

# we cannot customize the path of this temporal directory...
RUN mkdir /.horovod && chmod 777 /.horovod

FROM blueoil-base AS blueoil-dev

# Copy blueoil test code to docker image
COPY tests /home/blueoil/tests

# Add permission for all users
RUN chmod 777 /home/blueoil
RUN chmod 777 /home/blueoil/tests

# Setup entrypoint
COPY docker/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]

WORKDIR /home/blueoil
