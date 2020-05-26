# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2018

FROM osrf/ros:eloquent-desktop
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
    python-dev \
    python3-pip \
    ipython \
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
RUN pip3 install -U \
  argcomplete \
  flake8 \
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
  setuptools

# install Fast-RTPS dependencies
RUN apt-get install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev

# install CycloneDDS dependencies
RUN apt-get install --no-install-recommends -y \
  libcunit1-dev

# package required for building rviz plugins, but not included in base image
RUN apt-get install -y ros-eloquent-rviz-visual-testing-framework

# install rqt tools
RUN apt-get install -y \
    ros-eloquent-python-qt-binding \
    ros-eloquent-qt-gui-core \
    ros-eloquent-rqt*

# install cycloneDDS RMW implementation
RUN apt-get install -y ros-eloquent-rmw-cyclonedds-cpp

##### OPENCV

RUN pip install opencv-contrib-python

#### TENSORFLOW

WORKDIR $HOME

# install tensorflow
#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
RUN pip install tensorflow

# tensorflow object detection dependencies from:
# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-prerequisites
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
RUN apt-get install -y \
  protobuf-compiler \
  python-pil \
  python-lxml \
  python-tk
RUN pip install \
  Cython \
  contextlib2 \
  jupyter \
  matplotlib

# get tensorflow models repository
# currently object detection is part of research models that are not available on releases, so we use master
# TODO: check v2.20 WIP object detection https://github.com/tensorflow/models/tree/v2.2.0/official/vision/detection
RUN git clone https://github.com/tensorflow/models.git /usr/local/lib/python3.6/dist-packages/tensorflow/models

# protobuf compilation
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow/models/research
RUN protoc object_detection/protos/*.proto --python_out=.

# optional install COCO API for COCO metrics
RUN git clone https://github.com/cocodataset/cocoapi.git $HOME/cocoapi
WORKDIR $HOME/cocoapi/PythonAPI
RUN rm Makefile
COPY scripts/coco/Makefile $HOME/cocoapi/PythonAPI/Makefile
RUN make
RUN cp -r pycocotools /usr/local/lib/python3.6/dist-packages/tensorflow/models/research

##### NETWORK MODELS

# get mobilenet model for image detection
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow/models/research/object_detection
RUN wget download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -q --show-progress --progress=bar:force 2>&1
RUN tar -xf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

# get inception model for image classification
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow/models/tutorial/image/imagenet
RUN wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz -q --show-progress --progress=bar:force 2>&1
RUN mkdir inception-2015-12-05
RUN tar -xf inception-2015-12-05.tgz -C inception-2015-12-05

##### ROS2 VISION OPEN CV

## the vision open cv package is currently not used due to bugs in its ROS 2 version
#RUN apt-get update && apt-get install -y libcanberra-gtk-module libboost-all-dev
#RUN mkdir -p $HOME/ros2_vision_opencv_ws/src
#WORKDIR $HOME/ros2_vision_opencv_ws/src
#RUN git clone -b ros2 https://github.com/ros-perception/vision_opencv.git

##### INSTALL ROS 2 TENSORFLOW

WORKDIR $HOME

RUN rosdep update

RUN git clone https://github.com/alsora/ros2-tensorflow $HOME/ros2-tensorflow
RUN mkdir -p $HOME/tf_ws/src
WORKDIR $HOME/tf_ws/src
RUN ln -s $HOME/ros2-tensorflow/ros2-tensorflow .
WORKDIR $HOME/tf_ws
RUN rosdep install --from-paths src --ignore-src --rosdistro eloquent -y
RUN /bin/bash -c 'source /opt/ros/eloquent/setup.sh; \
  colcon build'

#### SET ENVIRONMENT

WORKDIR $HOME

RUN echo ' \n\
echo "Sourcing ROS2 packages..." \n\
source /opt/ros/eloquent/setup.sh \n\
source $HOME/tf_ws/install/local_setup.sh' >> $HOME/.bashrc

RUN echo ' \n\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/slim' >> $HOME/.bashrc

RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc
