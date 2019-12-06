#!/bin/bash

set -e

case "$1" in
    "init" | "train" | "convert" | "predict")
        blueoil "$@"
        ;;
    "" | "--help")
        blueoil
        ;;
    *)
        exec "$@"
        ;;
esac
