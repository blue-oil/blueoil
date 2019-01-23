#!/usr/bin/env bash
port=${1:-8000}

# move to this script dir. (lmnet docs)
cd `dirname $0`

script_dir=`pwd`

# Add lmnet/lmnet dir to PYTHONPATH for better-apidoc bug.
# https://github.com/goerz/better-apidoc/issues/9
export PYTHONPATH=${script_dir}/../lmnet:$PYTHONPATH

# update source rst files
better-apidoc -t source/_templates/ -feo ./source ../lmnet/

# make html. run liveload web server
sphinx-autobuild ${script_dir}/source ${script_dir}/_build/html/ -p $port


