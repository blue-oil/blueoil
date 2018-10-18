#!/bin/sh

# Enable protobuf
export PATH="/opt/protobuf/bin:$PATH"
export LIBRARY_PATH="/opt/protobuf/lib:$LIBRARY_PATH"
export CPLUS_INCLUDE_PATH="/opt/protobuf/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="/opt/protobuf/lib:$LD_LIBRARY_PATH"

# Enable python3.6
source scl_source enable rh-python36

# For Docker
export LM_CXX_C11=/opt/rh/devtoolset-6/root/usr/bin/g++
export LM_DOCKER1=enable

# Enable Intel tools
export PATH=$PATH:/opt/intelFPGA_lite/17.1/quartus/sopc_builder/bin
export PATH=$PATH:/opt/intelFPGA_lite/17.1/quartus/bin
export PATH=$PATH:/opt/intelFPGA_lite/17.1/modelsim_ase/linux

#export PATH=$PATH:/opt/intelFPGA_lite/17.1/hls/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intelFPGA_lite/17.1/hls/host/linux64/lib

source /opt/intelFPGA_lite/17.1/hls/init_hls.sh

export QUARTUS_ROOTDIR=/opt/intelFPGA_lite/17.1/quartus
export SOCEDS_DEST_ROOT=/opt/intelFPGA/17.1/embedded
source $SOCEDS_DEST_ROOT/env.sh 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intelFPGA_lite/17.1/quartus/linux64

export PATH=$PATH:/opt/llvm/bin
