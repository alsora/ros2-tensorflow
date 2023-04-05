# Copyright 2020 Alberto Soragna. All Rights Reserved.
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

from ros2_tf_core import img_conversion as img_utils
from ros2_tf_core import models as models_utils
from ros2_tf_core.tensorflow_node import TensorflowNode
from tf_interfaces.srv import ImageClassification as ImageClassificationSrv
from vision_msgs.msg import ObjectHypothesis


class ClassificationNode(TensorflowNode):

    def __init__(self, tf_model, node_name):
        super().__init__(node_name)

        # ROS parameters
        self.num_predictions_p = self.declare_parameter('num_predictions', 5)

        # Prepare the Tensorflow network
        self.startup(tf_model)

        # Advertise info about the Tensorflow network
        self.publish_vision_info(tf_model)
        # Create ROS entities
        self.create_service(
            ImageClassificationSrv,
            'image_classification',
            self.handle_image_classification_srv)

    def startup(self, tf_model):

        if tf_model.save_load_format != models_utils.SaveLoadFormat.FROZEN_MODEL:
            raise ValueError('Classification node currently supports only FROZEN MODELS')

        # Load model
        model_path = tf_model.compute_model_path()
        self.graph, self.session = models_utils.load_frozen_model(model_path)
        self.get_logger().info('Load model completed!')

        # Define input tensor
        self.input_image_tensor = self.graph.get_tensor_by_name('DecodeJpeg:0')

        # Define output tensor
        self.output_softmax_tensor = self.graph.get_tensor_by_name('softmax:0')

        self.warmup()

    def classify(self, image_np):
        start_time = self.get_clock().now()

        scores = self.session.run(
                self.output_softmax_tensor,
                feed_dict={self.input_image_tensor: image_np})

        elapsed_time = self.get_clock().now() - start_time
        elapsed_time_ms = elapsed_time.nanoseconds / 1000000
        self.get_logger().debug(f'Image classification took: {elapsed_time_ms} milliseconds')

        # Get top indices from softmax
        scores = np.squeeze(scores)
        top_classes = scores.argsort()[-self.num_predictions_p.value:][::-1]

        output_dict = {}
        output_dict['classification_classes'] = top_classes
        output_dict['classification_scores'] = [scores[i] for i in top_classes]

        return output_dict

    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.classify(image_np)

        self.get_logger().info('Warmup completed! Ready to receive real images!')

    def handle_image_classification_srv(self, request, response):

        image_np = img_utils.image_msg_to_image_np(request.image)

        output_dict = self.classify(image_np)

        classes = output_dict['classification_classes']
        scores = output_dict['classification_scores']

        response.classification.header.stamp = self.get_clock().now().to_msg()
        response.classification.results = []
        for i in range(len(classes)):
            hypotesis = ObjectHypothesis()
            hypotesis.class_id = str(classes[i].item())
            hypotesis.score = scores[i].item()
            response.classification.results.append(hypotesis)

        return response
