# ros2-tensorflow

Use Tensorflow to load pretrained neural networks and perform inference through ROS2 interfaces.

<img src="/data/detection.png" alt="Rviz2 detection output" width="50%" height="50%"/>
The output can be directly visualized through Rviz

## Requirements

In order to build the `ros2-tensorflow` package, the following dependencies are required

 - [OpenCV Python](https://pypi.org/project/opencv-python/)
 - [ROS2 Eloquent](https://index.ros.org/doc/ros2/Installation/)
 - [Tensorflow](https://www.tensorflow.org/install/)
 - [Tensorflow Object Detection Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
 - [Vision Msgs](https://github.com/Kukanani/vision_msgs)

The provided Dockerfile contains an Ubuntu 18.04 environment with all the dependencies already installed.

To use the Dockerfile:

    $ git clone https://github.com/alsora/ros2-tensorflow.git
    $ cd ros2-tensorflow/docker
    $ bash build.sh
    $ bash run.sh

#### Package build

This section describes how to build `ros2-tensorflow` package.

Get the source code

    $ git clone https://github.com/alsora/ros2-tensorflow.git $HOME/ros2-tensorflow

Create a ROS2 workspace

    $ mkdir -p $HOME/tf_ws/src

Add sources to workspace

    $ cd $HOME/tf_ws/src
    $ ln -s $HOME/ros2-tensorflow/ros2-tensorflow .

Install dependencies using rosdep

    $ cd $HOME/tf_ws
    $ rosdep install --from-paths src --ignore-src --rosdistro eloquent -y

Build and install the `ros2-tensorflow` package

    $ colcon build
    $ source install/local_setup.sh

## Usage

#### Image Detection Task

Get a Tensorflow model for image detection from the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models) uncompress it and place it inside the Tensorflow Models Object Detection directory at `/usr/local/lib/python3.6/dist-packages/tensorflow/models/research/object_detection`.

For example

    $ cd /usr/local/lib/python3.6/dist-packages/tensorflow/models/research/object_detection
    $ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
    $ tar -xf ssd_mobilenet_v1_coco_2017_11_17.tar.gz

**IMPORTANT**:If you want to use a different model or you installed Tensorflow Models in a different place, make sure to edit the paths inside the nodes and build again the workspace.

Test the image detection service by running in separate terminals

    $ ros2 run tf_detection_py server
    $ ros2 run tf_detection_py client_test

Real time image detection using your laptop camera

    $ rviz2
    $ ros2 run tf_detection_py subscriber
    $ ros2 run image_tools cam2image --ros-args -p topic:=camera -p frequency:=5.0


#### Image Classification Task

Get a Tensorflow model for image classification.

For example

    $ mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/models/tutorial/image/imagenet
    $ cd /usr/local/lib/python3.6/dist-packages/tensorflow/models/tutorial/image/imagenet
    $ wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    $ tar -xf inception-2015-12-05.tgz

**IMPORTANT**:If you want to use a different model or you installed Tensorflow Models in a different place, make sure to edit the paths inside the nodes and build again the workspace.

Test the image classification service by running in separate terminals

    $ ros2 run tf_classification_py server
    $ ros2 run tf_classification_py client_test
