# Copyright 2020. Alberto Soragna. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import numpy as np
import tensorflow as tf

from ros2_tensorflow.node.qos import qos_profile_vision_info
from ros2_tensorflow.node.tensorflow_node import TensorflowNode
from ros2_tensorflow.utils import img_conversion as img_utils
from ros2_tensorflow.utils import load_models as load_utils

from tf_interfaces.srv import ImageClassification as ImageClassificationSrv
from vision_msgs.msg import ObjectHypothesis, VisionInfo

TENSORFLOW_DIR = os.path.dirname(tf.__file__)
TENSORFLOW_IMAGENET_DIR = os.path.join(TENSORFLOW_DIR, 'models/tutorial/image/imagenet')

MODEL_NAME = 'inception-2015-12-05'
PATH_TO_FROZEN_MODEL = os.path.join(os.path.join(TENSORFLOW_IMAGENET_DIR, MODEL_NAME), 'classify_image_graph_def.pb')
PATH_TO_LABELS = os.path.join(os.path.join(TENSORFLOW_IMAGENET_DIR, MODEL_NAME), 'imagenet_2012_challenge_label_map_proto.pbtxt')


class ClassificationNode(TensorflowNode):

    def __init__(self, node_name):
        super().__init__(node_name)

        self.vision_info_pub = self.create_publisher(VisionInfo, 'vision_info', qos_profile=qos_profile_vision_info)

        self.startup()

    def startup(self):
        # Load model
        self.graph, self.session = load_utils.load_frozen_model(PATH_TO_FROZEN_MODEL)
        self.get_logger().info('Load model completed!')

        # Define input and output Tensors for classification_graph
        self.image_tensor = self.graph.get_tensor_by_name('DecodeJpeg:0')
        self.softmax_tensor = self.graph.get_tensor_by_name('softmax:0')

        # Publish vision info message (published only once with TRANSIENT LOCAL durability)
        vision_info_msg = VisionInfo()
        vision_info_msg.method = "TensorFlow image classification network, trained on imagenet"
        vision_info_msg.database_location = PATH_TO_LABELS
        self.vision_info_pub.publish(vision_info_msg)

        self.warmup()

    def create_classification_server(self, topic_name):
        self.srv = self.create_service(ImageClassificationSrv, topic_name, self.handle_image_classification_srv)

    def classify(self, image_np):
        start_time = self.get_clock().now()

        predictions = self.session.run(
                self.softmax_tensor,
                feed_dict={self.image_tensor: image_np})

        elapsed_time = self.get_clock().now() - start_time
        self.get_logger().debug('Image classification took: %r milliseconds' % (elapsed_time.nanoseconds / 1000000))

        return predictions

    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.classify(image_np)

        self.get_logger().info('Warmup completed! Ready to receive real images!')

    def handle_image_classification_srv(self, request, response):

        image_np = img_utils.image_msg_to_image_np(request.image)

        # This variable contains the softmax scores
        predictions = self.classify(image_np)

        # Get top indices from softmax
        predictions = np.squeeze(predictions)
        k = 5
        top_k_predictions = predictions.argsort()[-k:][::-1]

        response.classification.header.stamp = self.get_clock().now().to_msg()
        response.classification.results = []
        for prediction in top_k_predictions:
            hypotesis = ObjectHypothesis()
            hypotesis.id = prediction.item()
            hypotesis.score = predictions[prediction].item()
            response.classification.results.append(hypotesis)

        return response
