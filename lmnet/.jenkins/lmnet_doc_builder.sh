#!/bin/bash -xe

echo "# build docker container"
docker-compose build docs > /dev/null

echo "# stop document server"
docker-compose rm -f -s docs

echo "# remake and run document server"
docker-compose run --rm -d docs ./docs/remake_docs.sh 8001
