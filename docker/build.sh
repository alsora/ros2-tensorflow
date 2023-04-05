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

docker pull osrf/ros:humble-desktop
docker build $CLEAN_CMD -t $IMG_NAME .
