FROM centos:centos6.8 AS builder

LABEL maintener="Takeda Koji <takeda@leapmind.io>"

########################################################
# The begining of builder image
########################################################

# basic software installation
RUN yum -y update --exclude=kernel* --exclude=centos* && \
    yum -y groupinstall "Development Tools" && \
    yum -y install wget && \
    yum -y install centos-release-scl-rh && \
    yum -y install epel-release && \
    yum -y install scl-utils && \
    yum -y install gcc-c++ && \
    yum -y install devtoolset-6-gcc-c++ && \
    yum -y install git zlib-devel openssl-devel && \
    yum -y install rh-python36-python && \
    yum -y install rh-python36-python-devel && \
    yum -y install libjpeg-devel cmake3 unzip && \
    yum -y install xterm && \
    yum -y install gperftools-libs && \
    yum -y install mesa-libGL && \
    yum -y install mesa-libGLU && \
    yum -y install webkitgtk && \
    yum -y install libyaml-devel && \
    yum -y install ncurses-devel-5.7-4.20090207.el6


#COPY download /root/download
# Copy "top directory of dlk" of host to "/root/dlk" of guest 
COPY . /root/dlk


RUN cd /root/dlk/docker && \
    mkdir work_tools && \
    mv Quartus-lite-17.1.0.590-linux.tar  work_tools && \
    mv SoCEDSSetup-17.1.0.590-linux.run work_tools && \
    cd work_tools && \
    tar xvf Quartus-lite-17.1.0.590-linux.tar && \
    ./setup.sh --mode unattended --installdir /opt/intelFPGA_lite/17.1 --accept_eula 1 --disable-components arria_lite,cyclone,cyclone10lp,max,max10 && \
    ./SoCEDSSetup-17.1.0.590-linux.run --mode unattended --installdir /opt/intelFPGA/17.1 --accept_eula 1 && \
    cd .. && \
    rm -rf /root/dlk/docker/work_tools

# Download and Compilation of protobuf
RUN mkdir /root/protobuf && \
    cd /root/protobuf && \
    wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.zip && \ 
    unzip protobuf-cpp-3.5.1.zip && \
    cd protobuf-3.5.1 && \
    mkdir /opt/protobuf && \
    ./configure --prefix=/opt/protobuf && \
    make -j16 && \
    make install && \
    ln -s /usr/bin/cmake3 /usr/local/bin/cmake

#### Install x-compiler with Python26 and gcc4.4.7
# USER=xcomp DIR=build_ctng,install_ctng,build_xc
RUN yum -y install gperf texinfo help2man ncurses-devel python-devel && \
    useradd xcomp && \
    mkdir /home/xcomp/build_ctng && \ 
    cp /root/dlk/docker/lm_cross_arm.config /home/xcomp && \
    chown -R xcomp:xcomp /home/xcomp

USER xcomp

RUN cd /home/xcomp/build_ctng && \
    wget http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.23.0.tar.bz2 && \
    tar xvjf crosstool-ng-1.23.0.tar.bz2 && \
    cd crosstool-ng-1.23.0 && \
    ./configure --prefix=/home/xcomp/install_ctng && \
    make -j16 && \
    make install && \
    mkdir /home/xcomp/build_xc && \
    cd /home/xcomp/build_xc && \
    cp ../lm_cross_arm.config .config && \
    unset LD_LIBRARY_PATH && \
    unset LIBRARY_PATH && \
    unset CPLUS_INCLUDE_PATH &&\ 
    /home/xcomp/install_ctng/bin/ct-ng build

USER root

RUN chown -R root:root /home/xcomp/x-tools && \
    mv /home/xcomp/x-tools /opt

# Compilation of LLVM. This is last because newer gcc is requred.
RUN mkdir /root/src_llvm && \
    cd /root/src_llvm && \
    svn co http://llvm.org/svn/llvm-project/llvm/tags/RELEASE_501/final/ llvm-5.0.1 && \
    cd llvm-5.0.1/tools && \
    svn co http://llvm.org/svn/llvm-project/cfe/tags/RELEASE_501/final/ clang && \
    mkdir /root/build_llvm && \
    source scl_source enable rh-python36 && \
    source scl_source enable devtoolset-6 && \
    cd /root/build_llvm && \
    cmake -G "Unix Makefiles" /root/src_llvm/llvm-5.0.1/ -DCMAKE_INSTALL_PREFIX=/opt/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD="ARM;X86" && \
    make -j16 && \
    make install


########################################################
# The begining of running image
########################################################
FROM centos:centos6.8

# basic software installation
RUN yum -y update --exclude=kernel* --exclude=centos* && \
    yum -y groupinstall "Development Tools" && \
    yum -y install centos-release-scl-rh && \
    yum -y install epel-release && \
    yum -y install scl-utils && \
    yum -y install gcc-c++ && \
    yum -y install devtoolset-6-gcc-c++ && \
    yum -y install git zlib-devel openssl-devel && \
    yum -y install rh-python36-python && \
    yum -y install rh-python36-python-devel && \
    yum -y install libjpeg-devel && \
    yum -y install cmake3 && \
    yum -y install unzip && \
    yum -y install libyaml-devel && \
    yum -y install ncurses-devel-5.7-4.20090207.el6


# For Intel Tools
RUN yum -y install glibc.i686 && \
    yum -y install glibc-devel.i686 && \
    yum -y install libX11.i686 && \
    yum -y install libXext.i686 && \
    yum -y install libXft.i686 && \
    yum -y install libgcc.i686 && \
    yum -y install libgcc.x86_64 && \
    yum -y install libstdc++.i686 && \
    yum -y install libstdc++-devel.i686 && \
    yum -y install ncurses-devel.i686 && \
    yum -y install qt.i686 qt-x11.i686 && \
    yum -y install xterm && \
    yum -y install gperftools-libs && \
    yum -y install mesa-libGL && \
    yum -y install mesa-libGLU && \
    yum -y install webkitgtk && \
    yum clean all

#RUN mkdir /root/dlk
COPY --from=builder /root/dlk/docker/enable_tools.sh /etc/profile.d
COPY --from=builder /opt/protobuf /opt/protobuf
COPY --from=builder /opt/intelFPGA_lite/17.1 /opt/intelFPGA_lite/17.1
COPY --from=builder /opt/intelFPGA/17.1 /opt/intelFPGA/17.1
COPY --from=builder /opt/x-tools /opt/x-tools
COPY --from=builder /root/dlk /root/dlk
COPY --from=builder /opt/llvm /opt/llvm
RUN ln -s /opt/x-tools/arm-unknown-linux-gnueabihf/bin/arm-unknown-linux-gnueabihf-g++ /usr/local/bin/arm-linux-gnueabihf-g++ 
RUN ln -s /usr/bin/cmake3 /usr/local/bin/cmake 

