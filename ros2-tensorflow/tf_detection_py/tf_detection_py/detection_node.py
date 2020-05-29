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

        self.MIN_SCORE_THRESHOLD = 0.5

        self.republish_image = republish_image
        if (self.republish_image):
            self.image_pub = self.create_publisher(ImageMsg, 'output_image', 10)

        self.publish_vision_info(tf_model)

        self.startup(tf_model)

    def startup(self, tf_model):

        # Load labels
        path_to_labels = tf_model.compute_label_path()
        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)

        # Load model
        self.graph, self.session = tf_model.load_model()
        self.get_logger().info('Load model completed!')

        # Define input and output Tensors for detection_graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.warmup()

    def create_detection_server(self, topic_name):
        self.create_service(ImageDetectionSrv, topic_name, self.handle_image_detection_srv)

    def create_detection_subscription(self, topic_name):
        self.create_subscription(ImageMsg, topic_name, self.image_detection_callback, 10)
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)

    def detect(self, image_np):
        start_time = self.get_clock().now()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        elapsed_time = self.get_clock().now() - start_time
        elapsed_time_ms = elapsed_time.nanoseconds / 1000000
        self.get_logger().debug('Image detection took: %r milliseconds' % elapsed_time_ms)

        return boxes, scores, classes, num

    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.detect(image_np)

        self.get_logger().info('Warmup completed! Ready to receive real images!')

    def create_image_msg_with_detections(self, image_np, boxes, scores, classes):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                self.category_index,
                use_normalized_coordinates=True,
                min_score_thresh=self.MIN_SCORE_THRESHOLD,
                line_thickness=8)

        img_msg = ImageMsg()

        img_msg.height = image_np.shape[0]
        img_msg.width = image_np.shape[1]
        img_msg.encoding = 'bgr8'
        img_msg.data = image_np.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height
        img_msg.header.frame_id = 'map'

        return img_msg

    def create_detections_msg(self, image_np, boxes, scores, classes, num):
        img_height = image_np.shape[0]
        img_width = image_np.shape[1]

        # remove additional dimension
        boxes = boxes[0]
        classes = classes[0]
        num = num[0]
        scores = scores[0]

        detections = Detection2DArray()

        detections.header.stamp = self.get_clock().now().to_msg()
        detections.detections = []
        for i in range(int(num)):
            if scores[i] < self.MIN_SCORE_THRESHOLD:
                break

            det = Detection2D()
            det.header = detections.header
            det.results = []
            detected_object = ObjectHypothesisWithPose()
            detected_object.id = int(classes[i].item())
            detected_object.score = scores[i].item()
            det.results.append(detected_object)

            # box is min y, min x, max y, max x
            # in normalized coordinates
            box = boxes[i]
            det.bbox.size_y = (box[2] - box[0]) * img_height
            det.bbox.size_x = (box[3] - box[1]) * img_width
            det.bbox.center.x = (box[1] + box[3]) * img_height / 2
            det.bbox.center.y = (box[0] + box[2]) * img_width / 2

            detections.detections.append(det)

        return detections

    def handle_image_detection_srv(self, request, response):

        image_np = img_utils.image_msg_to_image_np(request.image)

        boxes, scores, classes, num = self.detect(image_np)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, boxes, scores, classes)
            self.image_pub.publish(img_msg)

        response.detections = self.create_detections_msg(image_np, boxes, scores, classes, num)

        return response

    def image_detection_callback(self, img_msg):

        image_np = img_utils.image_msg_to_image_np(img_msg)

        boxes, scores, classes, num = self.detect(image_np)

        if (self.republish_image):
            img_msg = self.create_image_msg_with_detections(image_np, boxes, scores, classes)
            self.image_pub.publish(img_msg)

        detections_msg = self.create_detections_msg(image_np, boxes, scores, classes, num)
        self.detection_pub.publish(detections_msg)
