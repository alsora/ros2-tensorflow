#!/bin/bash
#
# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2020

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

IMG_NAME="ros2_tensorflow_gpu"

NETWORK_SETTINGS="--net=host --privileged"

# --runtime=nvidia
XSOCK=/tmp/.X11-unix
DISPLAY_SETTINGS="-e DISPLAY=$DISPLAY -v $XSOCK:$XSOCK -v $XAUTHORITY:/root/.Xauthority"

DEV_SETTINGS="-v $THIS_DIR/..:/root/ros2-tensorflow"

ENTRY_CMD="bash"

# Start Docker container
docker run -it --rm \
    $NETWORK_SETTINGS \
    $DISPLAY_SETTINGS \
    $DEV_SETTINGS \
    $IMG_NAME \
    $ENTRY_CMD
