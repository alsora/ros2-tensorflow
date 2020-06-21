^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package tf_classification_py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.0 (2020-06-21)
------------------
* rename ros2_tensorflow package to ros2_tf_core
* download images for simple tests instead of using hardcoded paths
* add support for saved models in tf_detection_py
* store tensorflow network output in dictionary format
* use ros params in tf nodes
* unify server and subscriber examples
* create examples directories under tf_classification_py and tf_detection_py
* create method for publishing vision info in base class TensorflowNode
* remove hardcoded tensorflow model names and create TensorflowModel wrapper class
* create publisher for vision info msg in tf nodes
* use int instead of strings for vision_msgs id filed
  the message use a string in master branch, but an int in the eloquent release
* add copyright notice to files
* use vision_msgs detection2darray in detection node
* add vision_msgs as dependency and use them for classification
* move load frozen model function to utility file
* fix opencv dependency in package.xml files
* use ros time instead of datetime in nodes
* remove unused import and typographic cleanup
* fix wrong tensorflow key in package xml files (`#1 <https://github.com/alsora/ros2-tensorflow/issues/1>`_)
* updated the whole repo use ubuntu 18.04, tensorflow 2.0 and ros2 eloquent
* Contributors: Soragna, Alberto
