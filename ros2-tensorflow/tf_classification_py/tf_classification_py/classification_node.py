import rclpy
from rclpy.node import Node

import tensorflow as tf
import os
import numpy as np
from datetime import datetime

from tf_interfaces.srv import ImageClassification as ImageClassificationSrv
from ros2_tensorflow.node.tensorflow_node import TensorflowNode
from ros2_tensorflow.utils import img_conversion

TENSORFLOW_DIR = os.path.dirname(tf.__file__)
TENSORFLOW_IMAGENET_DIR =  os.path.join(TENSORFLOW_DIR, 'models/tutorial/image/imagenet')

MODEL_NAME = 'inception-2015-12-05'
PATH_TO_FROZEN_MODEL = os.path.join(os.path.join(TENSORFLOW_IMAGENET_DIR, MODEL_NAME), 'classify_image_graph_def.pb')

class ClassificationNode(TensorflowNode):

    def __init__(self, node_name):
        super().__init__(node_name)

        self.startup()
        

    def startup(self):
        super().load_model(PATH_TO_FROZEN_MODEL)

        # Definite input and output Tensors for classification_graph
        self.image_tensor = self.graph.get_tensor_by_name('DecodeJpeg:0')
        self.softmax_tensor = self.graph.get_tensor_by_name('softmax:0')

        self.get_logger().info("load model completed!")

        self.warmup()
        

    def create_classification_server(self, topic_name):
        self.srv = self.create_service(ImageClassificationSrv, topic_name, self.handle_image_classification_srv)


    def classify(self, image_np):
        start_time = datetime.now()

        predictions = self.session.run(
                self.softmax_tensor,
                feed_dict={self.image_tensor: image_np})

        elapsed_time = datetime.now() - start_time
        self.get_logger().info("image classification took: %r milliseconds" % (elapsed_time.total_seconds() * 1000))

        return predictions


    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.classify(image_np)

        self.get_logger().info("warmup completed!")
        self.get_logger().info("ready to receive real images!")


    def handle_image_classification_srv(self, request, response):

        image_np = img_conversion.image_msg_to_image_np(request.image)

        predictions = self.classify(image_np)

        predictions = np.squeeze(predictions)
        k = 5
        top_k_predictions = predictions.argsort()[-k:][::-1]

        response.classification = int(top_k_predictions[0])

        return response
