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

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from ros2_tensorflow.node.tensorflow_node import TensorflowNode
from ros2_tensorflow.utils import img_conversion as img_utils

from sensor_msgs.msg import Image as ImageMsg
from tf_interfaces.srv import ImageDetection as ImageDetectionSrv
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class DetectionNode(TensorflowNode):

    def __init__(self, tf_model, node_name, republish_image=True):
        super().__init__(node_name)

        self.republish_image = republish_image
        # ROS parameters
        self.min_score_thresh_p = self.declare_parameter('min_score_thresh', 0.5)
        self.input_topic_p = self.declare_parameter('input_topic', 'image')

        # Prepare the Tensorflow network
        self.startup(tf_model)

        # Advertise info about the Tensorflow network
        self.publish_vision_info(tf_model)
        # Create ROS entities
        self.create_service(ImageDetectionSrv, 'image_detection', self.handle_image_detection_srv)
        self.create_subscription(ImageMsg, self.input_topic_p.value, self.image_detection_callback, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.image_pub = self.create_publisher(ImageMsg, 'detections_image', 10)

    def startup(self, tf_model):

        # Load labels
        path_to_labels = tf_model.compute_label_path()
        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)

        # Load model
        self.graph, self.session = tf_model.load_model()
        self.get_logger().info('Load model completed!')

        # Define input tensor
        self.input_image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Define output tensors
        self.output_tensor_dict = {}
        self.output_tensor_dict['detection_boxes'] = self.graph.get_tensor_by_name('detection_boxes:0')
        self.output_tensor_dict['detection_classes'] = self.graph.get_tensor_by_name('detection_classes:0')
        self.output_tensor_dict['detection_scores'] = self.graph.get_tensor_by_name('detection_scores:0')
        self.output_tensor_dict['num_detections'] = self.graph.get_tensor_by_name('num_detections:0')

        self.warmup()

    def detect(self, image_np):
        start_time = self.get_clock().now()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Perform the inference
        output_dict = self.session.run(
            self.output_tensor_dict,
            feed_dict={self.input_image_tensor: image_np_expanded})

        elapsed_time = self.get_clock().now() - start_time
        elapsed_time_ms = elapsed_time.nanoseconds / 1000000
        self.get_logger().debug('Image detection took: %r milliseconds' % elapsed_time_ms)

        # Reshape the tensors:
        # - squeeze to remove the batch dimension (since here we fed a single image)
        # - keep only the first num_detections elements
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections] for key, value in output_dict.items()}

        # Convert classes from float to int
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.uint8)

        return output_dict

    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.detect(image_np)

        self.get_logger().info('Warmup completed! Ready to receive real images!')

    def create_image_msg_with_detections(self, image_np, output_dict):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                min_score_thresh=self.min_score_thresh_p.value,
                line_thickness=8)

        img_msg = ImageMsg()

        img_msg.height = image_np.shape[0]
        img_msg.width = image_np.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.data = image_np.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height
        img_msg.header.frame_id = 'map'

        return img_msg

    def create_detections_msg(self, image_np, output_dict):
        img_height = image_np.shape[0]
        img_width = image_np.shape[1]

        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        detections = Detection2DArray()

        detections.header.stamp = self.get_clock().now().to_msg()
        detections.detections = []
        for i in range(len(boxes)):
            if scores[i] < self.min_score_thresh_p.value:
                break

            det = Detection2D()
            det.header = detections.header
            det.results = []
            detected_object = ObjectHypothesisWithPose()
            detected_object.id = classes[i].item()
            detected_object.score = scores[i].item()
            det.results.append(detected_object)

            # box is ymin, xmin, ymax, xmax in normalized coordinates
            box = boxes[i]
            det.bbox.size_y = (box[2] - box[0]) * img_height
            det.bbox.size_x = (box[3] - box[1]) * img_width
            det.bbox.center.x = (box[1] + box[3]) * img_height / 2
            det.bbox.center.y = (box[0] + box[2]) * img_width / 2

            detections.detections.append(det)

        return detections

    def handle_image_detection_srv(self, request, response):

        image_np = img_utils.image_msg_to_image_np(request.image)

        output_dict = self.detect(image_np)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, output_dict)
            self.image_pub.publish(img_msg)

        response.detections = self.create_detections_msg(image_np, output_dict)

        return response

    def image_detection_callback(self, img_msg):

        image_np = img_utils.image_msg_to_image_np(img_msg)

        output_dict = self.detect(image_np)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, output_dict)
            self.image_pub.publish(img_msg)

        detections_msg = self.create_detections_msg(image_np, output_dict)
        self.detection_pub.publish(detections_msg)
