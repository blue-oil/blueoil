#!/bin/bash

if [ -z "$(ls -A lmnet/third_party/coco)" ]; then
    echo "seems submodule is not initialized correctly. Please execute following command, before re-run this script."
    echo "git submodule update --init --recursive"
    exit 1
fi

DOCKER_IMAGE_NAME="$(id -un)_blueoil:local_build"
# build docker image
docker build -t ${DOCKER_IMAGE_NAME} --build-arg python_version="3.6.3" -f docker/Dockerfile .

if [ "$1" == "--dist" ]; then
    # build docker image for distributed training
    docker build -t ${DOCKER_IMAGE_NAME}_dist --build-arg base_docker_image="${DOCKER_IMAGE_NAME}" -f docker/dist.Dockerfile .
fi
