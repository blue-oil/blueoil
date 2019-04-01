FROM nvidia/cuda:10.0-cudnn7-devel

MAINTAINER masuda@leapmind.io

# TensorBoard
EXPOSE 6006

ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$äLD_LIBRARY_PATHå

RUN echo "deb http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-updates main restricted universe multiverse \n\
deb http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb-src http://ftp.jaist.ac.jp/ubuntu/ xenial-backports main restricted universe multiverse \n\
deb http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse \n\
deb-src http://security.ubuntu.com/ubuntu xenial-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    dpkg \
    procps \
    libarchive13 \
    libcurl3 \
    libexpat1 \
    libgcc1 \
    libjsoncpp1 \
    libc6 \
    libstdc++6 \
    zlib1g \
    libc-bin \
    debconf \
    python3-minimal \
    python3.5 \
    libpython3-stdlib \
    libpython3-dev \
    python3.5-dev \
    ca-certificates \
    python-pip-whl \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    python3-wheel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


ARG USE_PYTHON_3_NOT_2
ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    ${PYTHON}-dev \
    swig

# Install dependencies for Pillow, Scipy and matplotlib for display, and requirements for pyenv and pyenv virtualenv installation
RUN ${PIP} --no-cache-dir install \
    Pillow \
    h5py \
    keras_applications \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    && test "${USE_PYTHON_3_NOT_2}" -eq 1 && true || ${PIP} --no-cache-dir install \
    enum34

# install aarch64 cross compile environment
RUN apt-get update && apt-get install -y \
    perl-base=5.26.1-6ubuntu0.3 \
    perl-modules-5.26 \
    libperl5.26 \
    perl \
    libdpkg-perl \
    libconfig-inifiles-perl \
    libfile-which-perl \
    libxml-sax-base-perl \
    perl-modules-5.22 \
    libperl5.22 \
    dpkg-dev \
    libconfig-auto-perl \
    libio-string-perl \
    libyaml-perl \
    libdebian-dpkgcross-perl \
    libfile-homedir-perl \
    libxml-namespacesupport-perl \
    libxml-sax-perl \
    libxml-simple-perl \
#    perl-base=5.22.1-9 \
#    perl-base=5.26.1-6ubuntu0.3 \
#    perlapi-5.22.1 \
    libxml-libxml-perl \
    binutils=2.26.1-1ubuntu1~16.04.8 \
    binutils-aarch64-linux-gnu \
#    libxml-simple-perl \
    liblocale-gettext-perl \ 
    libconfig-auto-perl \
    libdebian-dpkgcross-perl \
    g++-5-aarch64-linux-gnu \
    gcc-5-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    dpkg-cross \
    crossbuild-essential-arm64
  
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

# Copy lmnet sources to docker image
COPY lmnet /home/blueoil/lmnet
WORKDIR /home/blueoil/lmnet

# Install requirements
RUN pip install -r gpu.requirements.txt
# In order to install blueoil requirements `prompt_toolkit==1.0.15`, uninstall prompt-toolkit v2.0 that depends on `pdb==0.10.2`.
RUN pip uninstall -y prompt-toolkit

# Build coco. It needs numpy.
COPY lmnet/third_party third_party
# https://github.com/cocodataset/cocoapi/blob/440d145a30b410a2a6032827c568cff5dc1d2abf/PythonAPI/setup.py#L2
RUN cd third_party/coco/PythonAPI && pip install -e .

# For development 
RUN apt-get update && apt-get install -y \
    x11-apps \
    imagemagick \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy dlk sources to docker image
COPY dlk /home/blueoil/dlk
# Install dlk
WORKDIR /home/blueoil/dlk
RUN PYTHONPATH=python/dlk python setup.py install

# Copy blueoil sources to docker image
COPY blueoil /home/blueoil/blueoil
COPY setup.py /home/blueoil/setup.py
COPY setup.cfg /home/blueoil/setup.cfg
# Install blueoil
WORKDIR /home/blueoil
RUN python setup.py install
# Enable to be written by all user
RUN chmod 777 /home/blueoil
RUN chmod 777 /home/blueoil/dlk

# Copy output template
COPY output_template /home/blueoil/output_template

# Set __init__.py for python modules
RUN touch lmnet/lmnet/__init__.py
