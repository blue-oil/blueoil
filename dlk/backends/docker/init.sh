export INTEL_FPGA_DIR=/intelFPGA
export QUARTUS_DIR=$INTEL_FPGA_DIR/quartus
export QSYS_DIR=$INTEL_FPGA_DIR/qsys
export VSIM_DIR=$INTEL_FPGA_DIR/modelsim_ase
export HLS_DIR=$INTEL_FPGA_DIR/hls

export PATH=$QUARTUS_DIR/bin:$QSYS_DIR/bin:$VSIM_DIR/bin:$PATH
export QSYS_ROOTDIR=$QUARTUS_DIR/sopc_builder/bin
export CPLUS_INCLUDE_PATH=/usr/lib/gcc/x86_64-redhat-linux/4.4.7:/usr/include/c++/4.4.7:/usr/include/c++/4.4.7/x86_64-redhat-linux

source $HLS_DIR/init_hls.sh

