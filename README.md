# ros2-tensorflow

Use Tensorflow to load pretrained neural networks and perform inference through ROS2 interfaces.

<img src="/data/detection.png" alt="Rviz2 detection output" width="50%" height="50%"/>
The output can be directly visualized through Rviz

## Requirements

In order to build the `ros2-tensorflow` package, the following dependencies are needed

Required dependencies:
 - [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)

Rosdep dependencies:
 - [OpenCV Python](https://pypi.org/project/opencv-python/)
 - [Tensorflow](https://www.tensorflow.org/install/)
 - [Vision Msgs](https://github.com/Kukanani/vision_msgs)

Optional dependencies:
 - [Tensorflow Object Detection Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) for object detection tasks
 - [TensorFlow-Slim](https://github.com/google-research/tf-slim) for object segmentation tasks

The provided Dockerfile contains an Ubuntu 18.04 environment with all the dependencies and this package already installed.

To use the Dockerfile:

    $ git clone https://github.com/alsora/ros2-tensorflow.git
    $ cd ros2-tensorflow/docker
    $ bash build.sh
    $ bash run.sh

## Build

This section describes how to build the `ros2-tensorflow` package and the required depenencies in case you are not using the provided Dockerfile.

Get the source code and create the ROS 2 workspace

    $ git clone https://github.com/alsora/ros2-tensorflow.git $HOME/ros2-tensorflow
    $ mkdir -p $HOME/tf_ws/src
    $ cd $HOME/tf_ws
    $ ln -s $HOME/ros2-tensorflow/ros2-tensorflow src

Install required dependencies using rosdep

    $ rosdep install --from-paths src --ignore-src --rosdistro humble -y

Install the Tensorflow Object Detection Models (optional).

    $ sudo apt-get install -y protobuf-compiler python3-lxml python-tk
    $ pip install --user Cython contextlib2 jupyter matplotlib Pillow
    $ pip3 install --upgrade protobuf==3.20.3
    $ PYTHONDIRNAME=`python3 --version | awk -F ' ' '{ print $2 }' | awk -F . '{ print "python" $1 "." $2 }'`
    $ sudo mkdir models; sudo chmod 757 models
    $ git clone https://github.com/tensorflow/models.git /usr/local/lib/$PYTHONDIRNAME/dist-packages/tensorflow/models
    $ cd /usr/local/lib/$PYTHONDIRNAME/dist-packages/tensorflow/models/research
    $ protoc object_detection/protos/*.proto --python_out=.
    $
    $ echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/$PYTHONDIRNAME/dist-packages/tensorflow/models/research" >> $HOME/.bashrc

Install Tensorflow Slim (optional)
    
    $ pip install tf_slim

Build and install the `ros2-tensorflow` package

    $ colcon build
    $ source install/local_setup.sh

## Usage

The basic usage consists in creating a ROS 2 node which loads a Tensorflow model and another ROS 2 node that acts as a client and receives the result of the inference.

It is possible to specify which model a node should load.
Note that if the model is specified via url, as it is by default, the first time the node is executed a network connection will be required in order to download the model.

#### Object Detection Task

Test the object detection server by running in separate terminals

    $ ros2 run tf_detection_py server
    $ ros2 run tf_detection_py client_test

Setup a real object detection pipeline using a stream of images coming from a ROS 2 camera node

    $ rviz2
    $ ros2 run tf_detection_py server
    $ ros2 run image_tools cam2image --ros-args -p frequency:=2.0

#### Image Classification Task

Test the image classification server by running in separate terminals

    $ ros2 run tf_classification_py server
    $ ros2 run tf_classification_py client_test

## Loading different models

The repository contains convenient APIs for loading Tensorflow models into the ROS 2 nodes.

Models are defined using the `ModelDescriptor` class, which contains all the information required for loading a model and performing inference on it.
It can either contain a path where the model can be found on the machine or an URL where the model can be downloaded the first time.

Different model formats are also supported, such as frozen models and saved models.

Some known supported models are already present as examples.
See [classification models](ros2-tensorflow/tf_classification_py/tf_classification_py/models.py) and [detection models](ros2-tensorflow/tf_detection_py/tf_detection_py/models.py)


The [Tensorflow models repository](https://github.com/tensorflow/models) contains many pretrained models that can be used.
For example, you can get additional Tensorflow model for object detection from the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models).
