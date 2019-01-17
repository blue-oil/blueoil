#!/bin/bash -xe

# select GPU:0
export CUDA_VISIBLE_DEVICES=0

echo "# build docker container"
./docker_build.sh

echo "# run test"
./blueoil_test.sh