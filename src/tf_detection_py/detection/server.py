import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from my_interfaces.srv import ImageDetection as ImageDetectionSrv
#import cv_bridge

import tensorflow as tf
import numpy as np

import cv2
from PIL import Image

TENSORFLOW_DIR = os.path.dirname(tf.__file__)
TENSORFLOW_RESEARCH_DIR =  os.path.join(TENSORFLOW_DIR, 'models/research')
TENSORFLOW_OBJECT_DETECTION_DIR = os.path.join(TENSORFLOW_RESEARCH_DIR, 'object_detection')

sys.path.append(TENSORFLOW_RESEARCH_DIR) 
sys.path.append(TENSORFLOW_OBJECT_DETECTION_DIR) 

from utils import label_map_util
from utils import visualization_utils as vis_util

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


class DetectionServer(Node):

    def __init__(self):
        super().__init__('server')
        self.srv = self.create_service(ImageDetectionSrv, 'image_detection', self.handle_classify_image_srv)
        self.pub = self.create_publisher(ImageMsg, 'output_image')

        self.load_model()

        self.warmup()

    def load_model(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        self.get_logger().info("load model completed!")



    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        self.get_logger().info("warmup completed!")


    def handle_classify_image_srv(self, request, response):

        a = datetime.now()

        img_msg = request.image

        n_channels  = 3
        dtype = 'uint8'
        img_buf = np.asarray(img_msg.data, dtype=dtype)

        image_np = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                        dtype=dtype, buffer=img_buf)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)

        # remove additional dimension
        classes = classes[0]
        scores = scores[0]
        num = num[0]


        response.detections = []
        for i in range(int(num)):

            if scores[i] < MIN_SCORE_THRESHOLD:
                break
            
            response.detections.append(int(classes[i]))

        b = datetime.now()
        c = b - a

        self.get_logger().info("handle_classify_image_srv took: %r" % c)

        img_msg = ImageMsg()

        img_msg.height = image_np.shape[0]
        img_msg.width = image_np.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.data = image_np.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height
        img_msg.header.frame_id = "world"

        self.pub.publish(img_msg)

        return response




def main(args=None):

    rclpy.init(args=args)

    node = DetectionServer()

    rclpy.spin(node)

    rclpy.shutdown()



if __name__ == '__main__':
    main()
