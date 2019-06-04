#!/bin/bash

NAME=$0 # Name of the script
BASE_DIR=$(dirname $0)
RUN_SCRIPT=${BASE_DIR}/blueoil.sh
TEST_RESULT=0
TEST_CONFIG_PREFIX=created_by_test_script
YML_CONFIG_FILE=$1
TIME_STAMP=$(date +%Y%m%d%H%M%S)
TMP_TEST_DIR=./tmp/tests/${TIME_STAMP}
if [ ! -d ${TMP_TEST_DIR} ]; then
    echo "Creating directory for test: ${TMP_TEST_DIR}"
    mkdir -p ${TMP_TEST_DIR} || exit 1
fi
TEST_LOG_NO=0
FAILED_TEST_NO=""

# list of avairable optimizers listed in blueoil_init.py
OPTIMIZSERS=("Momentum" "Adam")

function usage_exit(){
	echo ""
	echo "Usage"
	echo "${NAME} <YML_CONFIG_FILE(optional)>"
	exit 1
}

function clean_exit(){
    ls -d ./saved/${TEST_CONFIG_PREFIX}_* > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Move files created by tests to ${TMP_TEST_DIR}"
        mv ./saved/${TEST_CONFIG_PREFIX}_* ${TMP_TEST_DIR}
        echo "If you want to clean up files created by tests, you can run 'rm -rf ${TMP_TEST_DIR}'"
    fi
    exit $1
}

function assert(){
    if [ $1 -ne $2 ]; then
        echo "##############################"
        echo "ERROR result is $1 (expect $2)"
        echo "##############################"
        FAILED_TEST_NO="${FAILED_TEST_NO} ${TEST_LOG_NO}"
        TEST_RESULT=1
    else
        echo "OK!"
    fi
}

function get_yaml_param(){
    YML_KEY=$1
    YML_FILE=$2
    echo $(grep ${YML_KEY} ${YML_FILE} | awk '{print $2}')
}

function get_dataset_format_by_task(){
    case "$1" in
        "classification" )
            echo "caltech101 delta_mark";;
        "object_detection" )
            echo "openimagesv4 delta_mark";;
        "semantic_segmentation" )
            echo "camvid_custom";;
    esac
}

function @(){
    VALID_EXIT_STATUS=$1
    TEST_LOG_FILE=${TMP_TEST_DIR}/test_${TEST_LOG_NO}.log
    shift
    echo "### $@" | tee -a ${TEST_LOG_FILE}
    "$@" >> ${TEST_LOG_FILE} 2>&1
    assert $? ${VALID_EXIT_STATUS}
    TEST_LOG_NO=$((TEST_LOG_NO+1))
}

function show_error_log(){
    for LOG_NO in ${FAILED_TEST_NO}
    do
        echo "#################################################"
        echo "Erorr log : ${TMP_TEST_DIR}/test_${LOG_NO}.log"
        echo "#################################################"
        cat ${TMP_TEST_DIR}/test_${LOG_NO}.log
        echo ""
    done
}

echo ""
echo "# Basic tests"

function init_test(){
    TEST_CASE=$1
    TASK_TYPE_NUMBER=$2
    NETWORK_NUMBER=$4
    DATASET_FORMAT_NUMBER=$5
    ENABLE_DATA_AUGMENTATION=$6
    OPTIMIZER_NUMBER=$7
    TRAINING_DATASET_PATH=$8
    VALIDATION_DATASET_PATH=$9
    CONFIG_NAME=${TEST_CONFIG_PREFIX}_${TEST_CASE}
    TEST_YML_CONFIG_FILE=./tests/config/${TEST_CASE}.yml
    echo "## Test of ${TEST_CASE}"
    echo "### ${RUN_SCRIPT} init"
    if [ "${VALIDATION_DATASET_PATH}" == "" ]; then
        SET_VALIDATION_PATH="2"
        EXPECT_VALIDATION=""
    else
        SET_VALIDATION_PATH="1"
        EXPECT_VALIDATION="
        expect \"validation dataset path:\"
        send \"${VALIDATION_DATASET_PATH}\n\"
        "
    fi
    if [ "${ENABLE_DATA_AUGMENTATION}" == "n" ]; then
        QA_ENABLE_DATA_AUGMENTATION="
            expect \"enable data augmentation?\"
            send \"${ENABLE_DATA_AUGMENTATION}\n\"
        "
    else
        QA_ENABLE_DATA_AUGMENTATION="
            expect \"enable data augmentation?\"
            send \"Y\n\"
            expect \"Please choose augmentors:\"
            send \" \n\"
        "
    fi
    expect -c "
        set timeout 5
        spawn env LANG=C ${RUN_SCRIPT} init
        expect \"your model name ():\"
        send \"${CONFIG_NAME}\n\"
        expect \"choose task type\"
        send \"${TASK_TYPE_NUMBER}\n\"
        expect \"choose network\"
        send \"${NETWORK_NUMBER}\n\"
        expect \"choose dataset format\"
        send \"${DATASET_FORMAT_NUMBER}\n\"
        expect \"training dataset path:\"
        send \"${TRAINING_DATASET_PATH}\n\"
        expect \"set validataion dataset?\"
        send \"${SET_VALIDATION_PATH}\n\"
        ${EXPECT_VALIDATION}
        expect \"batch size (integer):\"
        send \"\b\b1\n\"
        expect \"image size (integer x integer):\"
        send \"\n\"
        expect \"how many epochs do you run training (integer):\"
        send \"\b\b\b1\n\"
        expect \"select optimizer:\"
        send \"${OPTIMIZER_NUMBER}\n\"
        expect \"initial learning rate:\"
        send \"\n\"
        expect \"choose learning rate setting(tune1 / tune2 / tune3 / fixed):\"
        send \"\n\"
        ${QA_ENABLE_DATA_AUGMENTATION}
        expect \"apply quantization at the first layer?:\"
        send \"\n\"
        expect \"Next step:\"
    " > /dev/null
    assert $? 0
    # Wait for complete ${RUN_SCRIPT} init
    sleep 1
    mv config/${CONFIG_NAME}.yml ${TMP_TEST_DIR}/
    YML_CONFIG_FILE=${TMP_TEST_DIR}/${CONFIG_NAME}.yml
    @ 0 diff ${YML_CONFIG_FILE} ${TEST_YML_CONFIG_FILE}
}

function basic_test(){
    if [ ! -f "${YML_CONFIG_FILE}" ]; then
        echo "ERROR: No such file : ${YML_CONFIG_FILE}"
        usage_exit
    fi

    @ 0 ${RUN_SCRIPT} train ${YML_CONFIG_FILE}

    EXPERIMENT_DIR=$(ls -td ./saved/${CONFIG_NAME}* | head -1)
    @ 0 ${RUN_SCRIPT} convert ${YML_CONFIG_FILE} ${EXPERIMENT_DIR}

    @ 0 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures ${TMP_TEST_DIR} ${EXPERIMENT_DIR}
}

function additional_test(){
    echo "# Additional tests (option arguments)"
    echo "## Specify OUTPUT_DIRECTORY"
    OUTPUT_DIR=${TMP_TEST_DIR}/saved_${TIME_STAMP}

    @ 0 ${RUN_SCRIPT} train ${YML_CONFIG_FILE} ${OUTPUT_DIR}
    @ 0 ls -td ${OUTPUT_DIR}

    echo "## Specify EXPERIMENT_ID"
    TIME_STAMP=$(date +%Y%m%d%H%M%S)

    @ 0 ${RUN_SCRIPT} train ${YML_CONFIG_FILE} ${OUTPUT_DIR} test_experiment
    @ 0 ls -td ${OUTPUT_DIR}/test_experiment

    echo "## Specify existing EXPERIMENT_ID for re-training"
    @ 0 ${RUN_SCRIPT} train ${YML_CONFIG_FILE} ${OUTPUT_DIR} test_experiment

    echo "## Specify CHECKPOINT_NO"
    @ 0 ${RUN_SCRIPT} convert ${YML_CONFIG_FILE} ${EXPERIMENT_DIR} 1

    @ 0 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures ${TMP_TEST_DIR} ${EXPERIMENT_DIR} 1

    echo "# Additional tests (invalid arguments)"
    echo "## Invalid config"
    INVALID_CONFIG=config/test.yaaaaaaaaaaaml

    @ 1 ${RUN_SCRIPT} train ${INVALID_CONFIG}

    @ 1 ${RUN_SCRIPT} convert ${INVALID_CONFIG} ${EXPERIMENT_DIR}

    @ 1 ${RUN_SCRIPT} predict ${INVALID_CONFIG} lmnet/tests/fixtures ${TMP_TEST_DIR} ${EXPERIMENT_DIR}

    echo "## Invalid directories"
    @ 1 ${RUN_SCRIPT} train ${YML_CONFIG_FILE} README.md

    @ 1 ${RUN_SCRIPT} convert ${YML_CONFIG_FILE} ${EXPERIMENT_DIR}_invalid

    @ 1 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures_invalid ${TMP_TEST_DIR} ${EXPERIMENT_DIR}

    @ 1 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures saved_invalid ${EXPERIMENT_DIR}

    @ 1 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures ${TMP_TEST_DIR} ${EXPERIMENT_DIR}_invalid

    echo "## Invalid CHECKPOINT_NO"
    @ 1 ${RUN_SCRIPT} convert ${YML_CONFIG_FILE} ${EXPERIMENT_DIR} 2

    @ 1 ${RUN_SCRIPT} predict ${YML_CONFIG_FILE} lmnet/tests/fixtures ${TMP_TEST_DIR} ${EXPERIMENT_DIR} 2
}

# main
trap 'show_error_log; clean_exit 1' 1 2 3 15

if [ "${YML_CONFIG_FILE}" == "" ]; then
    ADDITIONAL_TEST_FLAG=0
    TASK_TYPE_NUMBER=1
    ENABLE_DATA_AUGMENTATION="y"
    for TASK_TYPE in "classification" "object_detection" "semantic_segmentation"
    do
        DATASET_FORMAT_NUMBER=1
        for DATASET_FORMAT in $(get_dataset_format_by_task ${TASK_TYPE})
        do
            for TEST_CASE in "${DATASET_FORMAT}_${TASK_TYPE}" "${DATASET_FORMAT}_${TASK_TYPE}_has_validation"
            do
                TRAINING_DATASET_PATH=$(get_yaml_param train_path tests/config/${TEST_CASE}.yml)
                VALIDATION_DATASET_PATH=$(get_yaml_param test_path tests/config/${TEST_CASE}.yml)
                OPTIMIZER_NUMBER=$(($((${TASK_TYPE_NUMBER} % ${#OPTIMIZSERS[@]}))+1))
                init_test ${TEST_CASE} ${TASK_TYPE_NUMBER} 1 1 ${DATASET_FORMAT_NUMBER} ${ENABLE_DATA_AUGMENTATION} ${OPTIMIZER_NUMBER} ${TRAINING_DATASET_PATH} ${VALIDATION_DATASET_PATH}
                basic_test
                if [ ${ADDITIONAL_TEST_FLAG} -eq 0 ]; then
                    # Run additional test only once
                    additional_test
                    ADDITIONAL_TEST_FLAG=1
                fi
            done
            DATASET_FORMAT_NUMBER=$((DATASET_FORMAT_NUMBER+1))
        done
        TASK_TYPE_NUMBER=$((TASK_TYPE_NUMBER+1))
    done
else
    CONFIG_NAME=$(echo $(basename ${YML_CONFIG_FILE}) | sed 's/\..*$//')
    basic_test
    additional_test
fi

if [ ${TEST_RESULT} -eq 0 ]; then
    echo "#############################"
    echo "### All tests are passed! ###"
    echo "#############################"
    clean_exit 0
else
    echo "##############################"
    echo "### Some tests are failed. ###"
    echo "##############################"
    show_error_log
    clean_exit 1
fi

