# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2018

FROM osrf/ros:humble-desktop
LABEL maintainer="alberto dot soragna at gmail dot com"

# working directory
ENV HOME /root
WORKDIR $HOME

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# general utilities
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    gdb \
    vim \
    nano \
    python3-dev \
    python3-pip \
    unzip

# pip
RUN pip3 install --upgrade pip

#### ROS2 SETUP

# Locale options
RUN apt-get install -y locales
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# install development tools and ROS tools
RUN apt-get install -y \
  build-essential \
  cmake \
  python3-catkin-pkg-modules \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool

# install python packages
# specify a version of flake8 as the latest (3.8 is not compatible with ament)
RUN pip3 install -U \
  argcomplete \
  flake8==3.7 \
  flake8-blind-except \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  pytest-repeat \
  pytest-rerunfailures \
  pytest \
  pytest-cov \
  pytest-runner \
  setuptools \
  protobuf==3.20.3

# install Fast-RTPS dependencies
RUN apt-get install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev

# install CycloneDDS dependencies
RUN apt-get install --no-install-recommends -y \
  libcunit1-dev

#### TENSORFLOW

WORKDIR $HOME

# install tensorflow
#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
#RUN pip install tensorflow

# tensorflow object detection dependencies from:
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-prerequisites
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
RUN apt-get install -y \
  protobuf-compiler \
  python3-lxml \
  python-tk

RUN pip install \
  Cython \
  contextlib2 \
  jupyter \
  matplotlib \
  Pillow

# get tensorflow models repository
# currently object detection is part of research models that are not available on releases, so we use master
# TODO: check v2.20 WIP object detection https://github.com/tensorflow/models/tree/v2.2.0/official/vision/detection
RUN git clone https://github.com/tensorflow/models.git /usr/local/lib/python3.10/dist-packages/tensorflow/models

# protobuf compilation
WORKDIR /usr/local/lib/python3.10/dist-packages/tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# Tensorflow-Slim (required for image segmentation)
RUN pip install tf_slim

# optional install COCO API for COCO metrics
#RUN git clone https://github.com/cocodataset/cocoapi.git $HOME/cocoapi
#WORKDIR $HOME/cocoapi/PythonAPI
#RUN rm Makefile
#COPY scripts/coco/Makefile $HOME/cocoapi/PythonAPI/Makefile
#RUN make
#RUN cp -r pycocotools /usr/local/lib/python3.6/dist-packages/tensorflow/models/research
#RUN pip install pycocotools

##### ROS2 VISION OPEN CV

## the vision open cv package is currently not used due to bugs in its ROS 2 version
#RUN apt-get update && apt-get install -y libcanberra-gtk-module libboost-all-dev
#RUN mkdir -p $HOME/ros2_vision_opencv_ws/src
#WORKDIR $HOME/ros2_vision_opencv_ws/src
#RUN git clone -b ros2 https://github.com/ros-perception/vision_opencv.git

##### INSTALL ROS 2 TENSORFLOW

WORKDIR $HOME

RUN rosdep update
RUN apt-get update

RUN mkdir -p $HOME/tf_ws/src
ADD https://api.github.com/repos/alsora/ros2-tensorflow/git/refs/heads/master version.json
#RUN git clone https://github.com/alsora/ros2-tensorflow $HOME/tf_ws/src/ros2-tensorflow
RUN git clone https://github.com/beechwoods-software/ros2-tensorflow $HOME/tf_ws/src/ros2-tensorflow
WORKDIR $HOME/tf_ws
RUN rosdep install --from-paths src --ignore-src --rosdistro humble -y
RUN /bin/bash -c 'source /opt/ros/humble/setup.sh; \
  colcon build'

#### SET ENVIRONMENT

WORKDIR $HOME

RUN echo ' \n\
echo "Sourcing ROS2 packages..." \n\
source /opt/ros/humble/setup.sh \n\
source $HOME/tf_ws/install/local_setup.sh' >> $HOME/.bashrc

RUN echo ' \n\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/dist-packages/tensorflow/models/research' >> $HOME/.bashrc

RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc
