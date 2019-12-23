FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04 AS base

LABEL maintainer="admin@blueoil.org"

# TensorBoard
EXPOSE 6006

# Python version settings
ARG python_version="3.6.3"

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda-10.0 \
    PYENV_ROOT=/usr/local/pyenv

# Add source list entry for GCC 9
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 60C317803A41BA51845E371A1E9377A2BA9EF27F
RUN echo 'deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main' | tee /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-xenial.list

RUN apt-get update && apt-get install -y \
    cmake \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-pil \
    libjpeg8-dev \
    zlib1g-dev \
    python3-matplotlib \
    liblapack-dev \
    git \
    make \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    g++-9 \
    crossbuild-essential-arm64 \
    g++-5-arm-linux-gnueabihf \
    openssh-client \
    openssh-server \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/arm-linux-gnueabihf-g++-5 /usr/bin/arm-linux-gnueabihf-g++

# Install OpenSSH for MPI to communicate between containers
RUN mkdir -p /var/run/sshd

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Install pyenv and virtualenv to deal with different python versions
# This is to avoid InvocationError which occurs at running tox test by changing python version with pyenv global/local command
# See also https://github.com/pyenv/pyenv-virtualenv/issues/202
RUN git clone https://github.com/yyuu/pyenv.git $PYENV_ROOT && \
    git clone https://github.com/yyuu/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv

# Pyenv and pyenv virtualenv environment settings
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)"

# Setup python virtualenv
RUN pyenv install ${python_version} && \
    pyenv virtualenv -p python${python_version%.*} ${python_version} python${python_version%.*} && \
    pyenv global python${python_version%.*}

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

RUN pip install -U pip setuptools

FROM base AS blueoil-base
# Install requirements
COPY ./*requirements.txt /home/blueoil/
RUN pip install -r /home/blueoil/gpu.requirements.txt

# Build coco. It needs numpy.
COPY lmnet/third_party /home/blueoil/lmnet/third_party
# https://github.com/cocodataset/cocoapi/blob/440d145a30b410a2a6032827c568cff5dc1d2abf/PythonAPI/setup.py#L2
RUN cd /home/blueoil/lmnet/third_party/coco/PythonAPI && pip install -e .

# Set env to install horovod with nccl and tensorflow option
ENV HOROVOD_GPU_ALLREDUCE=NCCL \
    HOROVOD_WITH_TENSORFLOW=1

# Install requirements for distributed training temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    pip install -r /home/blueoil/dist.requirements.txt && \
    ldconfig

# we cannot customize the path of this temporal directory...
RUN mkdir /.horovod && chmod 777 /.horovod

WORKDIR /home/blueoil

FROM blueoil-base AS blueoil-dev

# Copy blueoil sources to docker image
COPY blueoil /home/blueoil/blueoil
COPY setup.* /home/blueoil/
COPY output_template /home/blueoil/output_template
# Install blueoil

RUN python setup.py install
RUN chmod 777 /home/blueoil

# Copy dlk sources to docker image
COPY dlk /home/blueoil/dlk
# Install dlk
WORKDIR /home/blueoil/dlk
RUN PYTHONPATH=python/dlk python setup.py install
RUN chmod 777 /home/blueoil/dlk

# Copy lmnet sources to docker image
COPY lmnet /home/blueoil/lmnet

# Copy blueoil test code to docker image
COPY tests /home/blueoil/tests

ENV PYTHONPATH $PYTHONPATH:/home/blueoil:/home/blueoil/lmnet:/home/blueoil/dlk/python/dlk

# enable blueoil command
RUN pyenv rehash

# Setup entrypoint
COPY docker/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]

WORKDIR /home/blueoil
