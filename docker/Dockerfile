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

##### OPENCV3

RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"

RUN apt-get update && apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libjasper1 libdc1394-22-dev

RUN wget -O $HOME/opencv.zip https://github.com/opencv/opencv/archive/3.4.4.zip -q --show-progress --progress=bar:force 2>&1
RUN wget -O $HOME/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.4.zip -q --show-progress --progress=bar:force 2>&1

RUN unzip -q $HOME/opencv.zip
RUN unzip -q $HOME/opencv_contrib.zip
RUN mv $HOME/opencv-3.4.4 $HOME/opencv
RUN mv $HOME/opencv_contrib-3.4.4 $HOME/opencv_contrib
RUN rm $HOME/opencv.zip $HOME/opencv_contrib.zip

RUN mkdir $HOME/opencv/build
WORKDIR $HOME/opencv/build
COPY scripts/build_opencv.sh $HOME/scripts/build_opencv.sh
RUN chmod +x $HOME/scripts/build_opencv.sh
RUN /bin/bash -c 'source $HOME/.bashrc; \
  $HOME/scripts/build_opencv.sh'

#### TENSORFLOW SETUP

WORKDIR $HOME

# install tensorflow
#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
RUN pip install tensorflow

# general python visualization packages
RUN apt-get install -y \
  python3-tk \
  libgtk2.0-dev
RUN pip install matplotlib opencv-python

#### TENSORFLOW OBJECT DETECTION

# tensorflow object detection dependencies
RUN apt-get install -y \
  protobuf-compiler \
  python-pil \
  python-lxml \
  python-tk
RUN pip install \
  Cython \
  contextlib2 \
  jupyter

# get models
RUN git clone https://github.com/tensorflow/models.git /usr/local/lib/python3.6/dist-packages/tensorflow/models
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow/models
COPY scripts/label_map_util.patch /usr/local/lib/python3.6/dist-packages/tensorflow/models/label_map_util.patch
RUN git apply label_map_util.patch

# build coco api
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR $HOME/cocoapi/PythonAPI
RUN rm Makefile
COPY scripts/coco/Makefile $HOME/cocoapi/PythonAPI/Makefile
RUN make
RUN cp -r pycocotools /usr/local/lib/python3.6/dist-packages/tensorflow/models/research
WORKDIR /usr/local/lib/python3.6/dist-packages/tensorflow/models/research
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip -q --show-progress --progress=bar:force 2>&1
RUN unzip -q protobuf.zip
RUN /bin/bash -c './bin/protoc object_detection/protos/*.proto --python_out=.'

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

RUN apt-get install -y libcanberra-gtk-module libboost-all-dev

RUN mkdir -p $HOME/ros2_vision_opencv_ws/src
WORKDIR $HOME/ros2_vision_opencv_ws/src
RUN git clone -b ros2 https://github.com/ros-perception/vision_opencv.git

#### SET ENVIRONMENT

WORKDIR $HOME

RUN echo ' \n\
echo "Sourcing ROS2 packages..." \n\
source /opt/ros/eloquent/setup.sh' >> $HOME/.bashrc

RUN echo ' \n\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research:/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/slim' >> $HOME/.bashrc

RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc
