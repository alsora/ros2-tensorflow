#!/bin/bash
#
# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2018 

IMG_NAME="ros2_tensorflow_gpu"

if [[ $1 == "--force" ]]; then
    CLEAN_CMD="--no-cache"
else
    CLEAN_CMD=""
fi

docker build -t $CLEAN_CMD $IMG_NAME .