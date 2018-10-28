

from my_interfaces.srv import ImageClassification as ImageClassificationSrv

import rclpy
from rclpy.node import Node

import tensorflow as tf
import os
import numpy as np
from datetime import datetime

TENSORFLOW_DIR = os.path.dirname(tf.__file__)
TENSORFLOW_IMAGENET_DIR =  os.path.join(TENSORFLOW_DIR, 'models/tutorials/image/imagenet')

#sys.path.append(TENSORFLOW_IMAGENET_DIR)

MODEL_NAME = 'inception-2015-12-05'


PATH_TO_FROZEN_MODEL = os.path.join(os.path.join(TENSORFLOW_IMAGENET_DIR, MODEL_NAME), 'classify_image_graph_def.pb')


class ClassificationServer(Node):

    def __init__(self):
        super().__init__('server')
        self.srv = self.create_service(ImageClassificationSrv, 'image_classification', self.handle_classify_image_srv)

        self.load_model()

        self.warmup()

    def load_model(self):
        self.classification_graph = tf.Graph()
        with self.classification_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.Session(graph=self.classification_graph)

        # Definite input and output Tensors for classification_graph
        self.image_tensor = self.classification_graph.get_tensor_by_name('DecodeJpeg:0')
        self.softmax_tensor = self.classification_graph.get_tensor_by_name('softmax:0')

        self.get_logger().info("load model completed!")



    def warmup(self):

        image_np = np.uint8(np.random.randint(0, 255, size=(480, 640, 3)))

        # Actual detection.
        predictions = self.session.run(
                self.softmax_tensor,
                feed_dict={self.image_tensor: image_np})

        self.get_logger().info("warmup completed!")


    def handle_classify_image_srv(self, request, response):

        a = datetime.now()

        img_msg = request.image

        n_channels  = 3
        dtype = 'uint8'
        img_buf = np.asarray(img_msg.data, dtype=dtype)

        if n_channels == 1:
            image_np = np.ndarray(shape=(img_msg.height, img_msg.width),
                            dtype=dtype, buffer=img_buf)
        else:
            image_np = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_buf)


        # Actual detection.
        predictions = self.session.run(
                self.softmax_tensor,
                feed_dict={self.image_tensor: image_np})

        predictions = np.squeeze(predictions)

        k = 5
        top_k_predictions = predictions.argsort()[-k:][::-1]

        response.classification = int(top_k_predictions[0])

        b = datetime.now()
        c = b - a

        self.get_logger().info("handle_classify_image_srv took: %r" % c)


        return response




def main(args=None):

    rclpy.init(args=args)

    node = ClassificationServer()

    rclpy.spin(node)

    rclpy.shutdown()



if __name__ == '__main__':
    main()

