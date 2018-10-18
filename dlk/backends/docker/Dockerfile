FROM centos:centos7

ENV WORKSPACE=/workspace
ENV TMP_DIR=$WORKSPACE/tmp 
ENV INTEL_FPGA_VERSION=Quartus-lite-18.0.0.614 
ENV INTEL_FPGA_INSTALLER=$WORKSPACE/$INTEL_FPGA_VERSION-linux.tar 
ENV INTEL_FPGA_INSTALL_DIR=/intelFPGA

COPY . $WORKSPACE

RUN yum -y update && \
    yum -y groupinstall "Development Tools" && \
    yum -y install \
        epel-release \
        compat-gcc-44 compat-gcc-44-c++ \
        glibc.i686 glibc-devel.i686 libX11.i686 libXext.i686 libXft.i686 ncurses-devel.i686 which

RUN set -x && \
    cd $WORKSPACE && \
    mkdir $TMP_DIR && \
    tar xvf $INTEL_FPGA_INSTALLER -C $TMP_DIR && \
    mkdir $INTEL_FPGA_INSTALL_DIR && \
    $TMP_DIR/setup.sh --mode unattended --installdir $INTEL_FPGA_INSTALL_DIR --accept_eula 1 --disable-components arria_lite,cyclone,cyclone10lp,max,max10 && \
    cd / && \
    rm -rf $WORKSPACE
