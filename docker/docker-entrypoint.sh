#!/bin/bash

set -eu

case "$1" in
    "init")
        blueoil init "${@:2}"
        ;;
    "train")
        blueoil train "${@:2}"
        ;;
    "convert")
        blueoil convert "${@:2}"
        ;;
    "predict")
        blueoil predict "${@:2}"
        ;;
    *)
        exec "$@"
        ;;
esac
