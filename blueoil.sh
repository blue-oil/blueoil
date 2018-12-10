#!/bin/bash

function usage_exit(){
	echo ""
	echo "Usage"
	echo "${NAME} init"
	echo "${NAME} train <YML_CONFIG_FILE> <OUTPUT_DIRECTORY(optional)> <EXPERIMENT_ID(optional)>"
	echo "${NAME} convert <YML_CONFIG_FILE> <EXPERIMENT_DIRECTORY> <CHECKPOINT_NO(optional)>"
	echo "${NAME} predict <YML_CONFIG_FILE> <INPUT_DIRECTORY> <OUTPUT_DIRECTORY> <EXPERIMENT_DIRECTORY> <CHECKPOINT_NO(optional)>"
	echo "${NAME} tensorboard <EXPERIMENT_DIRECTORY> <PORT(optional)>"
	echo ""
	echo "EXAMPLE: Run training"
	echo "${NAME} train config/test.yml"
	echo ""
	echo "EXAMPLE: Convert trained model to binary files"
	echo "${NAME} convert config/test.yml ./saved/test_20180101000000"
	echo ""
	echo "EXAMPLE: Predict by using trained model"
	echo "${NAME} predict config/test.yml ./images ./result ./saved/test_20180101000000"
	echo ""
	echo "EXAMPLE: Serve tensorboard"
	echo "${NAME} tensorboard ./saved/test_20180101000000 6006"
	exit 1
}

function error_exit(){
	if [ $1 -ne 0 ]; then
		echo "ERROR: $2"
		exit 1
	fi
}

function check_num_args(){
	# Check if the number of args is valid
	if [ $1 $2 $3 ]
	then
		echo "ERROR: Invalid number of arguments"
		usage_exit
	fi
}

function check_files_and_directories(){
	# Check if file and directory are exist
	for x in "$@"
	do
		if [ ! -e ${x} ]; then
			echo "ERROR: No such file or directory : ${x}"
			usage_exit
		fi
	done
}

function create_directory(){
	if [ ! -d $1 ]; then
		echo "Directory does not exist, creating $1"
		mkdir -p $1
		error_exit $? "Can not create directory: $1"
	fi
}

function get_abs_path(){
	echo $(cd $1 && pwd)
	return 0
}

function get_yaml_param(){
	YML_KEY=$1
	YML_FILE=$2
	echo $(grep ${YML_KEY} ${YML_FILE} | awk '{print $2}')
}

# Set variables
NAME=$0 # Name of the script
BASE_DIR=$(dirname $0)
ABS_BASE_DIR=$(get_abs_path ${BASE_DIR})

# Docker image of blueoil
DOCKER_IMAGE=$(id -un)_blueoil:local_build

# Argument of path for docker needs to be absolute path.
GUEST_HOME_DIR="/home/blueoil"
GUEST_CONFIG_DIR="${GUEST_HOME_DIR}/config"
GUEST_OUTPUT_DIR="${GUEST_HOME_DIR}/saved"

# User IDs
USER_ID=$(id -u)
GROUP_ID=$(id -g)
# Shared docker options
PYHONPATHS="-e PYTHONPATH=/home/blueoil:/home/blueoil/lmnet:/home/blueoil/dlk/python/dlk"
SHARED_DOCKER_OPTIONS="--rm -t -u ${USER_ID}:${GROUP_ID} ${PYHONPATHS}"

# Mount source code directories if they are exist on host.
# Docker's -v option overwrite container's directory with mounted host directory.
# Currently, we do not mount dlk, because dlk directory include pre compiled library
if [ -e lmnet ] && [ -e blueoil ] ; then
	touch lmnet/lmnet/__init__.py
	SHARED_DOCKER_OPTIONS=${SHARED_DOCKER_OPTIONS}" \
		-v ${ABS_BASE_DIR}/lmnet:${GUEST_HOME_DIR}/lmnet \
		-v ${ABS_BASE_DIR}/blueoil:${GUEST_HOME_DIR}/blueoil"
fi

function blueoil_init(){
	CONFIG_DIR=${ABS_BASE_DIR}/config
	create_directory ${CONFIG_DIR}

	echo "#### Generate config ####"
	docker run ${SHARED_DOCKER_OPTIONS} -i \
		-v ${CONFIG_DIR}:${GUEST_CONFIG_DIR} ${DOCKER_IMAGE} \
		/bin/bash -c \
		"python blueoil/blueoil_init.py && mv *.yml ${GUEST_CONFIG_DIR}"
	error_exit $? "Failed to generate config"

	YML_CONFIG_FILE=$(ls -t ${CONFIG_DIR}/*.yml | head -1)
	echo "Config file is generated in ${YML_CONFIG_FILE}"
	echo "Next step: ${NAME} train ${YML_CONFIG_FILE}"
}

function set_variables_from_config(){
	CONFIG_DIR=$(dirname $1)
	CONFIG_DIR=$(get_abs_path ${CONFIG_DIR})
	YML_CONFIG_FILE_NAME=$(basename $1)
	YML_CONFIG_FILE=${CONFIG_DIR}/${YML_CONFIG_FILE_NAME}
	CONFIG_NAME=$(echo ${YML_CONFIG_FILE_NAME} | sed 's/\..*$//')
	PY_CONFIG_FILE_NAME=${CONFIG_NAME}.py
	DATASET_DIR=$(get_yaml_param train_path ${YML_CONFIG_FILE})
	VALIDATION_DATASET_DIR=$(get_yaml_param test_path ${YML_CONFIG_FILE})
	if [ "${VALIDATION_DATASET_DIR}" == "" ]; then
		VALIDATION_DATASET_DIR=${DATASET_DIR}
	fi
	check_files_and_directories ${DATASET_DIR} ${VALIDATION_DATASET_DIR}
	DATASET_ABS_DIR=$(get_abs_path ${DATASET_DIR})
	VALIDATION_DATASET_ABS_DIR=$(get_abs_path ${VALIDATION_DATASET_DIR})
	if [ "${DATASET_DIR}" == "${DATASET_ABS_DIR}" ] && [ "${VALIDATION_DATASET_DIR}" == "${VALIDATION_DATASET_ABS_DIR}" ]; then
		GUEST_DATA_DIR="/"
	elif [ "${DATASET_DIR}" != "${DATASET_ABS_DIR}" ] && [ "${VALIDATION_DATASET_DIR}" != "${VALIDATION_DATASET_ABS_DIR}" ]; then
		GUEST_DATA_DIR=$(get_abs_path .)
	else
		error_exit 1 "Training and validataion dataset are different type of path (one is Abusolute, another is Relative)"
	fi
}

function set_lmnet_docker_options(){
	LMNET_DOCKER_OPTIONS="${SHARED_DOCKER_OPTIONS} --runtime=nvidia \
		-v ${CONFIG_DIR}:${GUEST_CONFIG_DIR} \
		-v ${DATASET_ABS_DIR}:${DATASET_ABS_DIR} -e DATA_DIR=${GUEST_DATA_DIR} \
		-v ${VALIDATION_DATASET_ABS_DIR}:${VALIDATION_DATASET_ABS_DIR} \
		-v ${OUTPUT_DIR}:${GUEST_OUTPUT_DIR} -e OUTPUT_DIR=${GUEST_OUTPUT_DIR} \
		-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
}

function blueoil_train(){
	set_variables_from_config $1
	OUTPUT_DIR=${2:-./saved}
	create_directory ${OUTPUT_DIR}
	OUTPUT_DIR=$(get_abs_path ${OUTPUT_DIR})
	EXPERIMENT_ID=$3
	set_lmnet_docker_options

	if [ -z "${EXPERIMENT_ID}" ]; then
		TIME_STAMP=$(date +%Y%m%d%H%M%S)
		EXPERIMENT_ID=${CONFIG_NAME}_${TIME_STAMP}
	fi

	echo "#### Run training (${EXPERIMENT_ID}) ####"

	docker run ${LMNET_DOCKER_OPTIONS} ${DOCKER_IMAGE} \
		python blueoil/blueoil_train.py -c ${GUEST_CONFIG_DIR}/${YML_CONFIG_FILE_NAME} -i ${EXPERIMENT_ID}
	error_exit $? "Training exited with a non-zero status"

	if [ ! -f ${OUTPUT_DIR}/${EXPERIMENT_ID}/checkpoints/checkpoint ]; then
		error_exit 1 "Checkpoints are not created in ${OUTPUT_DIR}/${EXPERIMENT_ID}"
	fi
	echo "Checkpoints are created in ${OUTPUT_DIR}/${EXPERIMENT_ID}"
	echo "Next step: ${NAME} convert ${YML_CONFIG_FILE} ${OUTPUT_DIR}/${EXPERIMENT_ID}"
}

function set_variables_for_restore(){
	set_variables_from_config $1
	EXPERIMENT_DIR=$(get_abs_path $2)
	OUTPUT_DIR=$(dirname ${EXPERIMENT_DIR})
	EXPERIMENT_ID=$(basename ${EXPERIMENT_DIR})
	CHECKPOINT_NO=${3:-0}
	set_lmnet_docker_options

	if [ ${CHECKPOINT_NO} -gt 0 ]; then
		RESTORE_OPTION="--restore_path ${GUEST_OUTPUT_DIR}/${EXPERIMENT_ID}/checkpoints/save.ckpt-${CHECKPOINT_NO}"
		if [ ! -f ${OUTPUT_DIR}/${EXPERIMENT_ID}/checkpoints/save.ckpt-${CHECKPOINT_NO}.index ]; then
			error_exit 1 "Invalid number of checkpoint, there is no checkpoints ${OUTPUT_DIR}/${EXPERIMENT_ID}/checkpoints/save.ckpt-${CHECKPOINT_NO}"
		fi
	fi
}

function blueoil_convert(){
	set_variables_for_restore $1 $2 $3

	echo "#### Generate output files ####"

	docker run ${LMNET_DOCKER_OPTIONS} ${DOCKER_IMAGE} \
		python blueoil/blueoil_convert.py -i ${EXPERIMENT_ID} ${RESTORE_OPTION}
	error_exit $? "Failed to generate output files"

	# Set path for DLK
	DLK_DIR=$(ls -td ${EXPERIMENT_DIR}/export/*/* | head -1)
	OUTPUT_PROJECT_DIR=${DLK_DIR}/output
	DLK_OUTPUT_DIR=${OUTPUT_PROJECT_DIR}/models/lib

	if [ ! -f ${DLK_OUTPUT_DIR}/lib_x86.so ] || [ ! -f ${DLK_OUTPUT_DIR}/lib_arm.so ] || [ ! -f ${DLK_OUTPUT_DIR}/lib_fpga.so ] || [ ! -f ${DLK_OUTPUT_DIR}/libdlk_x86.a ] || [ ! -f ${DLK_OUTPUT_DIR}/libdlk_arm.a ] || [ ! -f ${DLK_OUTPUT_DIR}/libdlk_fpga.a ]; then
		error_exit 1 "Binary files are not generated in ${DLK_OUTPUT_DIR}"
	fi
	echo "Output files are generated in ${OUTPUT_PROJECT_DIR}"
	echo "Please see ${OUTPUT_PROJECT_DIR}/README.md to run prediction"
}

function blueoil_predict(){
	set_variables_for_restore $1 $4 $5
	PREDICT_INPUT_DIR=$(get_abs_path $2)
	PREDICT_OUTPUT_DIR=$(get_abs_path $3)

	echo "#### Predict from images ####"
	docker run ${LMNET_DOCKER_OPTIONS} \
		-v ${PREDICT_INPUT_DIR}:${PREDICT_INPUT_DIR} \
		-v ${PREDICT_OUTPUT_DIR}:${PREDICT_OUTPUT_DIR} ${DOCKER_IMAGE} \
		python lmnet/executor/predict.py -in ${PREDICT_INPUT_DIR} -o ${PREDICT_OUTPUT_DIR} -i ${EXPERIMENT_ID} ${RESTORE_OPTION}
	error_exit $? "Failed to predict from images"

	echo "Result files are created: ${PREDICT_OUTPUT_DIR}"
}


function blueoil_tensorboard(){
	echo "#### Serve tensorboard ####"
	EXPERIMENT_DIR=$(get_abs_path $1)
	PORT=${2:-6006}

	OUTPUT_DIR=$(dirname ${EXPERIMENT_DIR})
	EXPERIMENT_ID=$(basename ${EXPERIMENT_DIR})

	LMNET_DOCKER_OPTIONS="${SHARED_DOCKER_OPTIONS} --runtime=nvidia \
		-v ${OUTPUT_DIR}:${GUEST_OUTPUT_DIR} \
		-e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

	TENSORBOARD_DIR=${GUEST_OUTPUT_DIR}/${EXPERIMENT_ID}

	docker run ${LMNET_DOCKER_OPTIONS} \
		-i \
		-p ${PORT}:${PORT} \
		${DOCKER_IMAGE} \
		tensorboard --logdir ${TENSORBOARD_DIR} --host 0.0.0.0 --port ${PORT}
}


# Main
case "$1" in
	"init" )
		check_num_args $# -ne 1
		blueoil_init
		exit 0;;
	"train" )
		check_num_args $# -lt 2
		check_num_args $# -gt 4
		check_files_and_directories $2
		blueoil_train $2 $3 $4
		exit 0;;
	"convert" )
		check_num_args $# -lt 3
		check_num_args $# -gt 4
		check_files_and_directories $2 $3
		blueoil_convert $2 $3 $4
		exit 0;;
	"predict" )
		check_num_args $# -lt 5
		check_num_args $# -gt 6
		check_files_and_directories $2 $3 $4 $5
		blueoil_predict $2 $3 $4 $5 $6
		exit 0;;
	"tensorboard" )
		check_num_args $# -lt 2
		check_num_args $# -gt 3
		check_files_and_directories $2
		blueoil_tensorboard $2 $3
		exit 0;;
	* )
		echo "ERROR: Unsupported Operation."
		usage_exit
esac
