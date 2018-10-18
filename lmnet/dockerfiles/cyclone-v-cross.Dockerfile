FROM buildpack-deps:16.04 AS tensorflow-build

MAINTAINER wakisaka@leapmind.io

RUN apt-get update

# Build bazel
RUN apt-get install -y \
    bash-completion \
    openjdk-8-jdk \
    openjdk-8-jre-headless \
    zip \
    unzip \
    swig

# Build tensorflow
RUN apt-get install -y \
    python3 \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-wheel

# alias python=python3, pip=pip3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip setuptools


# Add libpython3-all-dev:armhf for cross compile.
RUN dpkg --add-architecture armhf
RUN echo 'deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ xenial main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
RUN echo 'deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ xenial-updates main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
RUN echo 'deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ xenial-security main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
RUN echo 'deb [arch=armhf] http://ports.ubuntu.com/ubuntu-ports/ xenial-backports main restricted universe multiverse' >> /etc/apt/sources.list.d/armhf.list
RUN sed -i 's#deb http://archive.ubuntu.com/ubuntu/#deb [arch=amd64] http://archive.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
RUN sed -i 's#deb http://security.ubuntu.com/ubuntu/#deb [arch=amd64] http://security.ubuntu.com/ubuntu/#g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y libpython3-all-dev:armhf


COPY ./third_party/tensorflow-on-arm /third_party/tensorflow-on-arm
WORKDIR /third_party/tensorflow-on-arm/build_tensorflow/

RUN chmod +x build_tensorflow.sh
COPY ./third_party/override/tensorflow-on-arm/build_tensorflow/configs/cyclone_v.conf cyclone_v.conf
RUN WORKDIR=/tmp/tensorflow-on-arm ./build_tensorflow.sh cyclone_v.conf



################################
# install qemu-user-static
################################
FROM buildpack-deps:16.04 AS qemu
RUN apt-get update && apt-get install -y \
    qemu-user-static \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*




################################
# cross build arm
################################
FROM arm32v7/buildpack-deps:16.04

COPY --from=qemu /usr/bin/qemu-arm-static /usr/bin/qemu-arm-static

RUN apt-get update && apt-get install -y \
    locales\
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Locale setting
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y \
    # Pillow and matplotlib has many dependencies for display.
    python3-pil \
    libjpeg8-dev \
    zlib1g-dev \
    python3-matplotlib \
    # for accelarat numpy
    libatlas-base-dev \
    libblas-dev \
    liblapack-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# alias python=python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --no-cache-dir -U pip setuptools

# Build Tensorflow

# install libstdc++6 for the tensorflow
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python-software-properties \
    && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y \
    libstdc++6 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY --from=tensorflow-build /tmp/tensorflow_pkg/tensorflow-1.9.0-cp35-none-any.whl /tmp/tensorflow-1.9.0-cp35-none-any.whl

# numpy 1.15 install error
RUN pip install --no-cache-dir numpy==1.14.0

RUN cd tmp && \
    pip install --no-cache-dir tensorflow-1.9.0-cp35-none-any.whl && \
    rm tensorflow-1.9.0-cp35-none-any.whl

COPY requirements.txt /tmp/requirements.txt

WORKDIR /home/lmnet

# install coco by source
COPY third_party third_party
RUN cd third_party/coco/PythonAPI && pip install --no-cache-dir -e .

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY lmnet lmnet
COPY executor executor
COPY configs configs
