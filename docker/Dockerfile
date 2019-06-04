FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

MAINTAINER masuda@leapmind.io

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
    cmake \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies for Pillow, Scipy and matplotlib for display, and requirements for pyenv and pyenv virtualenv installation
RUN apt-get update && apt-get install -y \
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
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install aarch64 cross compile environment
RUN apt-get update && apt-get install -y crossbuild-essential-arm64

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ENV PYENV_ROOT /usr/local/pyenv

# Install pyenv to deal with different python versions
RUN git clone https://github.com/yyuu/pyenv.git $PYENV_ROOT

# Install pyenv virtualenv
# This is to avoid InvocationError which occurs at running tox test by changing python version with pyenv global/local command
# See also https://github.com/pyenv/pyenv-virtualenv/issues/202
RUN git clone https://github.com/yyuu/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv

# Pyenv and pyenv virtualenv environment settings
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN eval "$(pyenv init -)" && \
    eval "$(pyenv virtualenv-init -)"

# Python version settings
ARG python_version="3.6.3"

# Setup python virtualenv
RUN pyenv install ${python_version} && \
    pyenv virtualenv -p python${python_version%.*} ${python_version} python${python_version%.*} && \
    pyenv global python${python_version%.*}

RUN pip install -U pip setuptools

# Install x-compiler
RUN apt-get update && apt-get install -y g++-5-arm-linux-gnueabihf && \
    ln -s /usr/bin/arm-linux-gnueabihf-g++-5 /usr/bin/arm-linux-gnueabihf-g++

# Install requirements
COPY lmnet/*requirements.txt /tmp/requirements/
RUN pip install -r /tmp/requirements/gpu.requirements.txt && rm -rf /tmp/requirements
# In order to install blueoil requirements `prompt_toolkit==1.0.15`, uninstall prompt-toolkit v2.0 that depends on `pdb==0.10.2`.
RUN pip uninstall -y prompt-toolkit

# Build coco. It needs numpy.
COPY lmnet/third_party /home/blueoil/lmnet/third_party
# https://github.com/cocodataset/cocoapi/blob/440d145a30b410a2a6032827c568cff5dc1d2abf/PythonAPI/setup.py#L2
RUN cd /home/blueoil/lmnet/third_party/coco/PythonAPI && pip install -e .

# For development
RUN apt-get update && apt-get install -y \
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

ENV PYTHONPATH $PYTHONPATH:/home/blueoil:/home/blueoil/lmnet:/home/blueoil/dlk/python/dlk
WORKDIR /home/blueoil
