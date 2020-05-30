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

import numpy as np

from ros2_tensorflow.node.tensorflow_node import TensorflowNode
from ros2_tensorflow.utils import img_conversion as img_utils

from tf_interfaces.srv import ImageClassification as ImageClassificationSrv
from vision_msgs.msg import ObjectHypothesis


class ClassificationNode(TensorflowNode):

    def __init__(self, tf_model, node_name):
        super().__init__(node_name)

        # Prepare the Tensorflow network
        self.startup(tf_model)

        # ROS parameters
        self.num_predictions_p = self.declare_parameter('num_predictions', 5)

        # Advertise info about the Tensorflow network
        self.publish_vision_info(tf_model)
        # Create ROS entities
        self.create_service(ImageClassificationSrv, 'image_classification', self.handle_image_classification_srv)

    def startup(self, tf_model):

        # Load model
        self.graph, self.session = tf_model.load_model()
        self.get_logger().info('Load model completed!')

        # Define input and output Tensors for classification_graph
        self.image_tensor = self.graph.get_tensor_by_name('DecodeJpeg:0')
        self.softmax_tensor = self.graph.get_tensor_by_name('softmax:0')

        self.warmup()

    def classify(self, image_np):
        start_time = self.get_clock().now()

        predictions = self.session.run(
                self.softmax_tensor,
                feed_dict={self.image_tensor: image_np})

        elapsed_time = self.get_clock().now() - start_time
        elapsed_time_ms = elapsed_time.nanoseconds / 1000000
        self.get_logger().debug('Image classification took: %r milliseconds' % elapsed_time_ms)

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
        k = self.num_predictions_p.value
        top_k_predictions = predictions.argsort()[-k:][::-1]

        response.classification.header.stamp = self.get_clock().now().to_msg()
        response.classification.results = []
        for prediction in top_k_predictions:
            hypotesis = ObjectHypothesis()
            hypotesis.id = prediction.item()
            hypotesis.score = predictions[prediction].item()
            response.classification.results.append(hypotesis)

        return response
