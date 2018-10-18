#!/bin/bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

STEP=$1
GEN_TEST=$2

QUARTUS="blueoil_de10-nano.prj"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
USER_DIR=$(pwd)

ARGS=2
OPTION_NUMBER=5
NAME=$0

HLS_SIMULATION_ELF=hls_simulation.elf
HLS_SYNTHESIS_ELF=hls_synthesis.elf

MSG_HEADER="[DLK]"

if [ $# -eq 0 ] || [ $STEP -gt $OPTION_NUMBER ]
then
	echo "${MSG_HEADER} Help message."
	echo "Usage: ${NAME} <step>"
	echo "<step> - the first step to execute."
  	echo "  0 - HLS simulation."
  	echo "  1 - HLS synthesis."
  	echo "  2 - Qsys HDL generation"
  	echo "  3 - Quartus Compilation"
  	echo "  4 - BSP configuration."
	exit 0
fi

if [ $# -ne $ARGS ]
then
	echo "${MSG_HEADER} ERROR: Bad number of arguments. ${ARGS} expected."
	exit 1
fi


if [ $STEP -lt 1 ]
then
  	echo "${MSG_HEADER} Running step 0: HLS simulation."
  	cd $SCRIPT_DIR
 	if [ ! -f Makefile ]
  	then
    		echo "${MSG_HEADER} ERROR: No Makefile found."
    		exit 1
  	fi
        if $GEN_TEST; then
          	make clean
          	make hls_simulation -j4
        else
                make clean >>$SCRIPT_DIR/msgs.out 2>&1
          	make hls_simulation -j4 >>$SCRIPT_DIR/msgs.out 2>&1
        fi

  	if [ ! -f $HLS_SIMULATION_ELF ]
  	then
    		echo "${MSG_HEADER} ERROR: HLS simulation failed."
    		exit 1
  	fi

  	./$HLS_SIMULATION_ELF cls
  	if [[ $? != 0 ]]
  	then
    		echo "${MSG_HEADER} ERROR: HLS simulation resulted wrong."
   	 	exit 1
  	fi
  	cd $USER_DIR
fi


if [ $STEP -lt 2 ]
then
	echo "${MSG_HEADER} Running step 1: HLS synthesis."
	cd $SCRIPT_DIR
	if [ ! -f Makefile ]
	then
		echo "${MSG_HEADER} ERROR: No Makefile found."
		exit 1
	fi

        if $GEN_TEST; then
        	make clean
        	make hls_synthesis -j4
        else
                make clean >>$SCRIPT_DIR/msgs.out 2>&1
        	make hls_synthesis -j4 >>$SCRIPT_DIR/msgs.out 2>&1
        fi

  	if [ ! -f $HLS_SYNTHESIS_ELF ]
  	then
    		echo "${MSG_HEADER} ERROR: HLS synthesis failed."
    		exit 1
  	fi
	cd $USER_DIR
fi

if [ $STEP -lt 3 ]
then
	echo "${MSG_HEADER} Running step 2: Qsys set-up and HDL generation."
	cd $QUARTUS
	if [ -f soc_system.qsys ]
	then
		rm soc_system.qsys
	fi
#qsys-script --script=soc_system.tcl
        if $GEN_TEST; then
                qsys-script --search-path=$SCRIPT_DIR/hls/src/hls_operators.prj/components/binary_convolution_hls/,$ --script=soc_system.tcl
        else
                qsys-script --search-path=$SCRIPT_DIR/hls/src/hls_operators.prj/components/binary_convolution_hls/,$ --script=soc_system.tcl >>$SCRIPT_DIR/msgs.out 2>&1
        fi
	if [ -d soc_system ]
	then
		rm -rf soc_system
	fi
#qsys-generate soc_system.qsys --synthesis=VHDL
        if $GEN_TEST; then
                qsys-generate soc_system.qsys --synthesis=VHDL --search-path=$SCRIPT_DIR/hls/src/hls_operators.prj/components/binary_convolution_hls/,$
        else
                qsys-generate soc_system.qsys --synthesis=VHDL --search-path=$SCRIPT_DIR/hls/src/hls_operators.prj/components/binary_convolution_hls/,$ >>$SCRIPT_DIR/msgs.out 2>&1
        fi

	cd $USER_DIR
fi

if [ $STEP -lt 4 ]
then
	echo "${MSG_HEADER} Running step 3: Quartus compilation."
	cd $QUARTUS
	QPROJ_PATH=( ./*.qpf )
	QPROJ_EXT=$(basename $QPROJ_PATH)
	QPROJ=$(echo $QPROJ_EXT | cut -d"." -f1)
	echo "${MSG_HEADER} Found project: ${QPROJ}"

	# This is workaround for docker
	PREFIX_QUARTUS_SH=""
	if [ "$LM_DOCKER1" = 'enable' ]
	then
		echo "quartus_sh (LM_DOCKER1: enable)"
		PREFIX_QUARTUS_SH="LD_PRELOAD=/usr/lib64/libtcmalloc_minimal.so.4"
	fi
        
	if $GEN_TEST; then
                bash -c "$PREFIX_QUARTUS_SH quartus_sh --flow compile $QPROJ"
        else
                bash -c "$PREFIX_QUARTUS_SH quartus_sh --flow compile $QPROJ >>$SCRIPT_DIR/msgs.out 2>&1"
        fi
	SOF_FILE="output_files/${QPROJ}.sof"
	if [ ! -f $SOF_FILE ]
	then
		echo "${MSG_HEADER} ERROR: SOF file not found."
		exit 1
	fi
        if $GEN_TEST; then
                quartus_cpf -c -o bitstream_compression=on $SOF_FILE soc_system.rbf
        else
                quartus_cpf -c -o bitstream_compression=on $SOF_FILE soc_system.rbf >>$SCRIPT_DIR/msgs.out 2>&1
        fi
	if [ ! -d bootfiles ]
	then
		mkdir bootfiles
	fi
	mv soc_system.rbf bootfiles/
	cd $USER_DIR
fi

if [ $STEP -lt 5 ]
then
	echo "${MSG_HEADER} Running step 4: BSP generation."
	cd $QUARTUS

	if [ -d bsp ]
	then
		rm -rf bsp
	fi
	mkdir bsp
        if $GEN_TEST; then
                bsp-create-settings --type spl --bsp-dir bsp --settings settings.bsp --preloader-settings-dir hps_isw_handoff/soc_system_hps_0
        else
        	bsp-create-settings --type spl --bsp-dir bsp --settings settings.bsp --preloader-settings-dir hps_isw_handoff/soc_system_hps_0 >>$SCRIPT_DIR/msgs.out 2>&1
        fi
	cd bsp
        if $GEN_TEST; then
                make
        else
                make >>$SCRIPT_DIR/msgs.out 2>&1
        fi
	cd ..
	if [ ! -d bootfiles ]
	then
		mkdir bootfiles
	fi
	mv bsp/preloader-mkpimage.bin bootfiles
	cd $USER_DIR
fi

echo "${MSG_HEADER} OK!"
exit 0
