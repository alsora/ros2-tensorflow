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

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from ros2_tf_core import img_conversion as img_utils
from ros2_tf_core.tensorflow_node import TensorflowNode
from sensor_msgs.msg import Image as ImageMsg
from tf_detection_py.detection_models import create as create_detection_model
from tf_interfaces.srv import ImageDetection as ImageDetectionSrv
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class DetectionNode(TensorflowNode):

    def __init__(self, tf_model, node_name, republish_image=True):
        super().__init__(node_name)

        self.republish_image = republish_image
        # ROS parameters
        self.min_score_thresh_p = self.declare_parameter('min_score_thresh', 0.5)
        # Default disabled, use values smaller than 1 to enable
        self.ioa_thresh_p = self.declare_parameter('ioa_thresh', 1.0)
        self.input_topic_p = self.declare_parameter('input_topic', 'image')

        # Prepare the Tensorflow network
        self.startup(tf_model)

        # Advertise info about the Tensorflow network
        self.publish_vision_info(tf_model)
        # Create ROS entities
        self.create_service(ImageDetectionSrv, 'image_detection', self.handle_image_detection_srv)
        img_topic = self.input_topic_p.value
        self.create_subscription(ImageMsg, img_topic, self.image_detection_callback, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.image_pub = self.create_publisher(ImageMsg, 'detections_image', 10)

    def startup(self, tf_model):

        # Load labels
        path_to_labels = tf_model.compute_label_path()
        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)

        # Load model
        self.network_model = create_detection_model(tf_model)
        self.get_logger().info('Load model completed!')

        self.warmup()

    def filter_detections(self, output_dict):
        # Filter the detected objects to remove wrong detections
        # - Remove boxes with detection score too small
        # - Remove boxes according to Intersection Over Area
        # among boxes with same detected class
        to_be_removed = []

        boxes = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores = output_dict['detection_scores']

        for i in range(len(boxes)):
            if scores[i] < self.min_score_thresh_p.value:
                to_be_removed.append(i)
                continue

            area = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            for j in range(len(boxes)):
                if classes[j] != classes[i] or i == j or j in to_be_removed:
                    continue

                if scores[j] < self.min_score_thresh_p.value:
                    to_be_removed.append(j)
                    continue

                # determine the coordinates of the intersection box
                intersect_ymin = max(boxes[i][0], boxes[j][0])
                intersect_xmin = max(boxes[i][1], boxes[j][1])
                intersect_ymax = min(boxes[i][2], boxes[j][2])
                intersect_xmax = min(boxes[i][3], boxes[j][3])

                intersect_height = max(0, intersect_ymax - intersect_ymin)
                intersect_width = max(0, intersect_xmax - intersect_xmin)
                intersect_area = intersect_height * intersect_width

                if (intersect_area / area) > self.ioa_thresh_p.value:
                    to_be_removed.append(i)
                    if scores[i] > scores[j]:
                        self.get_logger().warn('Filter high score detection for low score one')
                    break

        # Remove all elements marked as "to_be_removed"
        output_dict['detection_boxes'] = np.delete(
            output_dict['detection_boxes'], to_be_removed, axis=0)
        output_dict['detection_classes'] = np.delete(
            output_dict['detection_classes'], to_be_removed, axis=0)
        output_dict['detection_scores'] = np.delete(
            output_dict['detection_scores'], to_be_removed, axis=0)
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = np.delete(
                output_dict['detection_masks'], to_be_removed, axis=0)

    def detect(self, image_np):
        start_time = self.get_clock().now()

        output_dict = self.network_model.inference(image_np)

        elapsed_time = self.get_clock().now() - start_time
        elapsed_time_ms = elapsed_time.nanoseconds / 1000000
        self.get_logger().debug(f'Image detection took: {elapsed_time_ms} milliseconds')

        self.filter_detections(output_dict)

        return output_dict

    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.detect(image_np)

        self.get_logger().info('Warmup completed! Ready to receive real images!')

    def create_image_msg_with_detections(self, image_np, output_dict):
        # Visualization of the results of a detection.
        # NOTE: this method modifies the provided image
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks', None),
                use_normalized_coordinates=True,
                min_score_thresh=self.min_score_thresh_p.value,
                line_thickness=8)

        img_msg = img_utils.image_np_to_image_msg(image_np)
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
            det = Detection2D()
            det.header = detections.header
            det.results = []
            detected_object = ObjectHypothesisWithPose()
            detected_object.id = str(classes[i].item())
            detected_object.score = scores[i].item()
            det.results.append(detected_object)

            # box is ymin, xmin, ymax, xmax in normalized coordinates
            box = boxes[i]
            det.bbox.size_y = (box[2] - box[0]) * img_height
            det.bbox.size_x = (box[3] - box[1]) * img_width
            det.bbox.center.x = (box[1] + box[3]) * img_height / 2
            det.bbox.center.y = (box[0] + box[2]) * img_width / 2

            if (self.republish_image):
                box_img = image_np[
                    int(box[0]*img_height):int(box[2]*img_height),
                    int(box[1]*img_width):int(box[3]*img_width)]

                det.source_img = img_utils.image_np_to_image_msg(box_img)

            detections.detections.append(det)

        return detections

    def handle_image_detection_srv(self, request, response):

        image_np = img_utils.image_msg_to_image_np(request.image)

        output_dict = self.detect(image_np)

        response.detections = self.create_detections_msg(image_np, output_dict)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, output_dict)
            self.image_pub.publish(img_msg)

        return response

    def image_detection_callback(self, img_msg):

        image_np = img_utils.image_msg_to_image_np(img_msg)

        output_dict = self.detect(image_np)

        detections_msg = self.create_detections_msg(image_np, output_dict)
        self.detection_pub.publish(detections_msg)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, output_dict)
            self.image_pub.publish(img_msg)
