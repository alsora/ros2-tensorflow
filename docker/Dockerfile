# @author Alberto Soragna (alberto dot soragna at gmail dot com)
# @2018

FROM ubuntu:16.04
MAINTAINER Alberto Soragna alberto dot soragna at gmail dot com

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
    vim \
    nano \
    python-dev \
    python3-pip \
    ipython

# pip
RUN pip3 install --upgrade pip


#### ROS2 SETUP


# Locale options
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# setup sources
RUN apt-get install -y lsb-release
RUN curl http://repo.ros2.org/repos.key | apt-key add -
RUN sh -c 'echo "deb [arch=amd64,arm64] http://repo.ros2.org/ubuntu/main `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'

# ROS setup requirements
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  python3-colcon-common-extensions \
  python3-pip \
  python-rosdep \
  python3-vcstool \
  wget

RUN python3 -m pip install -U \
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
  pytest-rerunfailures

# install additional ubuntu 16.04 requirements
RUN python3 -m pip install -U \
  pytest \
  pytest-cov \
  pytest-runner \
  setuptools

# install Fast-RTPS dependencies
RUN apt-get install --no-install-recommends -y \
  libasio-dev \
  libtinyxml2-dev \
  libopensplice67

# create ros2 sdk workspace
RUN mkdir -p $HOME/ros2_sdk/src
WORKDIR $HOME/ros2_sdk
RUN wget https://raw.githubusercontent.com/ros2/ros2/release-latest/ros2.repos
RUN vcs import src < ros2.repos

# initialize rosdep and install dependencies
RUN rosdep init
RUN rosdep update
RUN rosdep install --from-paths src --ignore-src --rosdistro bouncy -y --skip-keys "console_bridge fastcdr fastrtps libopensplice67 rti-connext-dds-5.3.1 urdfdom_headers"

# build the workspace
RUN colcon build --symlink-install


##### OPENCV3

RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

RUN git clone -b 3.4 https://github.com/Itseez/opencv.git $HOME/opencv
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
  python-tk \
  unzip
RUN pip install \
  Cython \
  contextlib2 \
  jupyter \
  matplotlib \
  opencv-python

# get models and coco api
RUN git clone https://github.com/tensorflow/models.git /usr/local/lib/python3.5/dist-packages/tensorflow/models
RUN git clone https://github.com/cocodataset/cocoapi.git

# build coco api
WORKDIR $HOME/cocoapi/PythonAPI
RUN rm Makefile
COPY scripts/coco/Makefile $HOME/cocoapi/PythonAPI/Makefile
RUN make
RUN cp -r pycocotools /usr/local/lib/python3.5/dist-packages/tensorflow/models/research
WORKDIR /usr/local/lib/python3.5/dist-packages/tensorflow/models/research
RUN wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
RUN unzip protobuf.zip
RUN /bin/bash -c './bin/protoc object_detection/protos/*.proto --python_out=.'



##### NETWORK MODELS

# get mobilenet model for image detection
WORKDIR /usr/local/lib/python3.5/dist-packages/tensorflow/models/research/object_detection
RUN wget download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
RUN tar -xf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

# get inception model for image classification
WORKDIR /usr/local/lib/python3.5/dist-packages/tensorflow/models/tutorial/image/imagenet
RUN wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
RUN mkdir inception-2015-12-05
RUN tar -xf inception-2015-12-05.tgz -C inception-2015-12-05

##### ROS2 WORKSPACE

RUN apt-get install -y libcanberra-gtk-module libboost-all-dev

RUN mkdir -p $HOME/ws/src
WORKDIR $HOME/ws/src
RUN git clone -b ros2 https://github.com/ros-perception/vision_opencv.git

#### SET ENVIRONMENT


WORKDIR $HOME

RUN echo ' \n\
echo "Sourcing ROS2 packages..." \n\
source $HOME/ros2_sdk/install/setup.sh' >> $HOME/.bashrc

RUN echo ' \n\
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research:/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/slim' >> $HOME/.bashrc

RUN echo ' \n\
alias python="python3"' >> $HOME/.bashrc

# add scripts folder to docker image
COPY scripts $HOME/scripts





