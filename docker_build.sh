#!/bin/bash

if [ -z "$(ls -A lmnet/third_party/coco)" ]; then
    echo "seems submodule is not initialized correctly. Please execute following command, before re-run this script."
    echo "git submodule update --init --recursive"
    exit 1
fi

docker build -t $(id -un)_blueoil:local_build --build-arg python_version="3.6.3" -f docker/Dockerfile .

