#!/bin/bash -xe

echo "# build docker container"
docker-compose build --pull --no-cache tensorflow > /dev/null

echo "# build coco PythonAPI"
docker-compose run --rm tensorflow bash -c "cd third_party/coco/PythonAPI && make -j8"

echo "# run test with python3.5"
docker-compose run --rm tensorflow tox -e py35
