#!/bin/bash -xe

echo "# build docker container"
./docker_build.sh

echo "# run test"
DOCKER_OPTION="-e PYTHONPATH=python/dlk -v /root/.ssh:/root/.ssh --net=host"
docker run --rm -t ${DOCKER_OPTION} $(id -un)_blueoil:local_build /bin/bash -c \
    "apt-get update && apt-get install -y iputils-ping && cd dlk && python setup.py test"

echo "# check PEP8"
docker run --rm -t $(id -un)_blueoil:local_build /bin/bash -c \
    "cd dlk && pycodestyle --ignore=W --max-line-length=120 --exclude='*static/pb*','*docs/*','*.eggs*','*tvm/*','*tests/*','backends/*' ."
