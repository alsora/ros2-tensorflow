import os
import sys
from datetime import datetime

import tensorflow as tf
import numpy as np

import cv2

TENSORFLOW_DIR = os.path.dirname(tf.__file__)
TENSORFLOW_RESEARCH_DIR = os.path.join(TENSORFLOW_DIR, 'models/research')
TENSORFLOW_OBJECT_DETECTION_DIR = os.path.join(TENSORFLOW_RESEARCH_DIR, 'object_detection')

sys.path.append(TENSORFLOW_RESEARCH_DIR) 
sys.path.append(TENSORFLOW_OBJECT_DETECTION_DIR) 

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from sensor_msgs.msg import Image as ImageMsg
from tf_interfaces.srv import ImageDetection as ImageDetectionSrv
from ros2_tensorflow.node.tensorflow_node import TensorflowNode
from ros2_tensorflow.utils import img_conversion
#import cv_bridge

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_MODEL = os.path.join(os.path.join(TENSORFLOW_OBJECT_DETECTION_DIR, MODEL_NAME), 'frozen_inference_graph.pb')
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(TENSORFLOW_OBJECT_DETECTION_DIR,"data/mscoco_label_map.pbtxt")

NUM_CLASSES = 90
MIN_SCORE_THRESHOLD = 0.5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class DetectionNode(TensorflowNode):

    def __init__(self, node_name, publish_bbox = True):
        super().__init__(node_name)

        self.publish_bbox = publish_bbox
        if (self.publish_bbox):
            self.pub = self.create_publisher(ImageMsg, 'output_image', 10)

        self.startup()
        

    def startup(self):
        super().load_model(PATH_TO_FROZEN_MODEL)

        # Definite input and output Tensors for detection_graph
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
        self.srv = self.create_service(ImageDetectionSrv, topic_name, self.handle_image_detection_srv)


    def create_detection_subscription(self, topic_name):
        self.sub = self.create_subscription(ImageMsg, topic_name, self.image_detection_callback, 10)


    def detect(self, image_np):
        start_time = datetime.now()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        elapsed_time = datetime.now() - start_time
        self.get_logger().debug("Image detection took: %r milliseconds" % (elapsed_time.total_seconds() * 1000))

        return boxes, scores, classes, num


    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        self.detect(image_np)

        self.get_logger().info("Warmup completed! Ready to receive real images!")


    def publish_detection_results(self, image_np, boxes, scores, classes):
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

        img_msg = ImageMsg()

        img_msg.height = image_np.shape[0]
        img_msg.width = image_np.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.data = image_np.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height
        img_msg.header.frame_id = "map"

        self.pub.publish(img_msg)


    def handle_image_detection_srv(self, request, response):

        image_np = img_conversion.image_msg_to_image_np(request.image)

        boxes, scores, classes, num = self.detect(image_np)

        if (self.publish_bbox):
            self.publish_detection_results(image_np, boxes, scores, classes)

        # remove additional dimension
        classes = classes[0]
        scores = scores[0]
        num = num[0]

        response.detections = []
        for i in range(int(num)):
            if scores[i] < MIN_SCORE_THRESHOLD:
                break
            response.detections.append(int(classes[i]))

        return response


    def image_detection_callback(self, img_msg):

        image_np = img_conversion.image_msg_to_image_np(img_msg)

        boxes, scores, classes, num = self.detect(image_np)

        if (self.publish_bbox):
            self.publish_detection_results(image_np, boxes, scores, classes)
