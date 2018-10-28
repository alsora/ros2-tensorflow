# ros2-tensorflow

Use Tensorflow to load pretrained neural networks and perform inference through ROS2 interfaces.

<img src="/data/detection.png" alt="Rviz2 detection output" width="50%" height="50%"/>
The output can be directly visualized through Rviz

## Requirements

 - [ROS2 Bouncy](https://index.ros.org/doc/ros2/Installation/)
 - [Tensorflow](https://www.tensorflow.org/install/)
 - [Tensorflow Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 - [OpenCV3](https://docs.opencv.org/3.0-beta/doc/tutorials/introduction/linux_install/linux_install.html)

The provided Dockerfile contains an Ubuntu 16.04 environment with all the dependencies already installed.

To use the Dockerfile:

    $ cd docker
    $ bash build.sh
    $ bash run.sh

## Build 

Get the source code

    $ git clone https://github.com/alsora/ros2-tensorflow.git $HOME/ros2-tensorflow

Create a ROS2 workspace

    $ mkdir -p ws/src

Add sources to workspace
    $ cd ws/src
    $ ln -s $HOME/ros2-tensorflow/src/* .

Build and source the workspace
    $ cd ..
    $ colcon build
    $ source install/local_setup.sh

## Usage

#### Image Detection Task

Get a Tensorflow model for image detection ([mobilenet_v1](download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz)), uncompress it and place it inside the Tensorflow Models Object Detection directory at `/usr/local/lib/python3.5/dist-packages/tensorflow/models/research/object_detection`.

**NOTE**:If you want to use a different model or you installed Tensorflow Models in a different place, make sure to edit the paths inside the nodes and build again the workspace.

Test the image detection service by running in separate terminals

    $ ros2 run tf_detection_py server
    $ ro2s run tf_detection_py client_test

Real time image detection using your laptop camera

    $ rviz2
    $ ros2 run tf_detection_py_subscriber
    $ ros2 run image_tools cam2image -t camera -f 15


#### Image Classification Task

Get a Tensorflow model for image classification ([inception network](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)), uncompress it and place it inside the Tensorflow Models Imagenet directory at `/usr/local/lib/python3.5/dist-packages/tensorflow/models/tutorial/image/imagenet`.

**NOTE**:If you want to use a different model or you installed Tensorflow Models in a different place, make sure to edit the paths inside the nodes and build again the workspace.

Test the image classification service by running in separate terminals

    $ ros2 run tf_classification_py server
    $ ro2s run tf_classification_py client_test
