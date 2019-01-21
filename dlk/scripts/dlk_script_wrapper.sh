#!/bin/bash

trap '
  trap - INT # restore default INT handler
  kill -s INT "$$"
' INT

# Get the absolute path from relative one.
function get_abs_path(){
	echo $(cd $1 && pwd)
	return 0
}

function category2dirname(){
	if [ $1 = "CLS" ]
	then
		echo "classification"
	elif [ $1 = "DET" ]
	then
		echo "object_detection"
	elif [ $1 = "SEG" ]
	then
		echo "segmentation"
	else
		echo "ERROR"	
	fi
	return 0
}

function put_help_message(){
	echo "${MSG_HEADER} Help message."
	echo "Usage: ${NAME} <CLS or DET or SEG> <INPUT_DIRECTORY> <OUTPUT_DIRECTORY>"
}

function make_each_target(){
	
	make clean "${quiet_args[@]}"
	FLAGS="-D__WITHOUT_TEST__" make "${1}" -j4 "${quiet_args[@]}"
	# mv "${2}" ./blueoil_de10-nano.prj/bootfiles/
	if [[ "${1}" = "lm_x86" ]]
	then
	    strip "${2}"
	elif [[ "${1}" = "lm_aarch64" ]]
	then
	    aarch64-linux-gnu-strip "${2}"  
	elif [[ "${1}" = "lm_arm" ]] || [[ "${1}" = "lm_fpga" ]]
	then
	    arm-linux-gnueabihf-strip "${2}"
	elif [[ "${1}" = "lib_x86" ]]
	then
	    strip -x --strip-unneeded "${2}"
	elif [[ "${1}" = "lib_aarch64" ]]
	then
	    aarch64-linux-gnu-strip -x --strip-unneeded "${2}"	    
	elif [[ "${1}" = "lib_arm" ]] || [[ "${1}" = "lib_fpga" ]]
	then
	    arm-linux-gnueabihf-strip -x --strip-unneeded "${2}"
	fi
	mv "${2}" "${3}"
}


NAME=$0 # Name of the script
CATEGORY_FPGA=$1 # CLS or DET or SEG


# Check if input directory exists (existing is OK)
if [ ! -r $2 ]
then
	echo "${MSG_HEADER} ERROR: No input directory found."
	exit 1
fi

# Check if output directory exist (existing is OK)
if [ ! -r $3 ]
then
	echo "${MSG_HEADER} ERROR: No output directory found."
	exit 1
fi

# Check if output directory is empty (empty is OK)
if [ -n "$(ls $3)" ]
then
	echo "${MSG_HEADER} ERROR: Output directory is NOT empty."
	exit 1
fi

INPUT_DATA_DIR=`get_abs_path $2`
OUTPUT_DATA_DIR=`get_abs_path $3`

REL_SCRIPTS_DIR=`dirname $0`
SCRIPTS_DIR=`get_abs_path ${REL_SCRIPTS_DIR}`
DLK_DIR=$SCRIPTS_DIR/..
TMP_DIR=$DLK_DIR/tmp

MSG_HEADER="[DLK_SCRIPT_WRAPPER]"
ARGS=3
STEP=0

# Whether to run/generate tests or not
GEN_TEST=false
#GEN_TEST=true

# Setup insertion of args for suppressing the debug msgs
args=()
[ $GEN_TEST == "true" ] && args+=( '-gts' )
args+=( '-hq' )
args+=( '-ts' )
args+=( '-cache' )

quiet_args=()
[ $GEN_TEST == "false" ] && quiet_args+=( '--quiet' )

#echo "<step> - the first step to execute."
#echo "	0 - pb conversion."
#echo "	1 - Quartus compilation."
#echo "	2 - elf files generation."

PRJ_NAME=this
PB_FILE=$INPUT_DATA_DIR/*.pb
PROJ_DIR=$DLK_DIR/tmp/$PRJ_NAME.prj/

# Check if number of arguments is valid or not.
if [ $# -ne $ARGS ]
then
	put_help_message
	exit 0
fi

# Check if the CATEGORY_FPGA is valid or not.
if [ `category2dirname $CATEGORY_FPGA` == "ERROR" ]
then
	put_help_message
	exit 0
fi

# 1. python convertion
if [ $STEP -lt 1 ]
then
	echo "${MSG_HEADER} Running step 0: pb conversion."
	cd $DLK_DIR

	if [ "$LM_DOCKER1" = 'enable' ]
	then
		echo "setup.py install (LM_DOCKER1: enable)"
		scl enable devtoolset-6 'PYTHONPATH=python/dlk python setup.py install'
	else
		echo "setup.py install (normal)"
                if $GEN_TEST; then
                        PYTHONPATH=python/dlk python setup.py install
                else
                        PYTHONPATH=python/dlk python setup.py install >>msgs.out 2>&1
                fi
	fi

        # Whether to run the python customtest or not
	if $GEN_TEST; then
		PYTHONPATH=python/dlk python setup.py test
	fi

	if [ ! -f $PB_FILE ]
	then
		echo "${MSG_HEADER} ERROR: No pb found."
		exit 1
	fi

        PYTHONPATH=python/dlk python3 python/dlk/scripts/generate_project.py -i $PB_FILE -p $PRJ_NAME -o $TMP_DIR "${args[@]}"

	if [ ! -r $PROJ_DIR ]
	then
		echo "${MSG_HEADER} ERROR: project generation failed."
		exit 1
	fi

fi

# 2. update and run build script
#if [ $STEP -lt 2 ]
#then
#	echo "${MSG_HEADER} Running step 1: Quartus compilation."
#	cd $PROJ_DIR
#	rm blueoil_build_altera.sh
#	cp $SCRIPTS_DIR/dlk_build_altera.sh .
#	./dlk_build_altera.sh 0 $GEN_TEST
#	if [[ $? -ne 0 ]] && exit
#	then
#		echo "${MSG_HEADER} ERROR: project generation failed."
#		exit 1
#	fi
#fi
if [ $STEP -lt 2 ]
then
	echo ${MSG_HEADER} "Running step 1: copying binaries of FPGA"
	cd $PROJ_DIR	
	CAT_DIRNAME=`category2dirname $CATEGORY_FPGA`
	mkdir ${OUTPUT_DATA_DIR}/${CAT_DIRNAME}
	cp $DLK_DIR/hw/intel/de10_nano/qconv_with_kn2row/preloader-mkpimage.bin ${OUTPUT_DATA_DIR}/${CAT_DIRNAME}
	cp $DLK_DIR/hw/intel/de10_nano/qconv_with_kn2row/soc_system.rbf ${OUTPUT_DATA_DIR}/${CAT_DIRNAME}
fi

# 3. prepare elf and other files
if [ $STEP -lt 3 ]
then
	echo "${MSG_HEADER} Running step 2: elf files generation."
	cd $PROJ_DIR
	
	#make clean "${quiet_args[@]}"
	#FLAGS="-D__WITHOUT_TEST__" make lm_fpga -j4 "${quiet_args[@]}"
	#mv lm_fpga.elf ./blueoil_de10-nano.prj/bootfiles/

	#make clean "${quiet_args[@]}"
	#FLAGS="-D__WITHOUT_TEST__" make lm_arm -j4 "${quiet_args[@]}"
	#mv lm_arm.elf ./blueoil_de10-nano.prj/bootfiles/
	
	make_each_target lm_x86 lm_x86.elf ${OUTPUT_DATA_DIR}
	make_each_target lm_arm lm_arm.elf ${OUTPUT_DATA_DIR}
	make_each_target lm_aarch64 lm_aarch64.elf ${OUTPUT_DATA_DIR}
	make_each_target lm_fpga lm_fpga.elf ${OUTPUT_DATA_DIR}
	make_each_target lib_x86 lib_x86.so ${OUTPUT_DATA_DIR}
	make_each_target lib_arm lib_arm.so ${OUTPUT_DATA_DIR}
	make_each_target lib_aarch64 lib_aarch64.so ${OUTPUT_DATA_DIR}
	make_each_target lib_fpga lib_fpga.so ${OUTPUT_DATA_DIR}
	make_each_target ar_x86 libdlk_x86.a ${OUTPUT_DATA_DIR}
	make_each_target ar_arm libdlk_arm.a ${OUTPUT_DATA_DIR}
	make_each_target ar_aarch64 libdlk_aarch64.a ${OUTPUT_DATA_DIR}
	make_each_target ar_fpga libdlk_fpga.a ${OUTPUT_DATA_DIR}
fi





echo "${MSG_HEADER} OK!"
exit 0
