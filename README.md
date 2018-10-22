# ros2-tensorflow

Train an image classification CNN and perform inference through a ROS2 service

## Requirements

 - ROS2 Bouncy
 - Tensorflow

The provided Dockerfile contains an already setup Ubuntu 16.04 environment.

## Build

Get the source code

    $ git clone https://github.com/alsora/ros2-tensorflow.git

Create a ROS2 workspace

    $ mkdir -p ws/src

Add sources to workspace

    $ ln -s ros2-tensorflow/src/* ws/src

Build and source the workspace

    $ colcon build
    $ source install/local_setup.sh

## Usage

Train a simple MNIST model
    $ cd ros2-tensorflow
    $ python create_mnist_model.py

Start a server node

    $ ros2 run python_tf server --model_dir models/mnist_model

Start a client node

    $ ros2 run python_tf client

