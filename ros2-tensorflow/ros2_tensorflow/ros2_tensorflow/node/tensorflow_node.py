
import rclpy
from rclpy.node import Node

import tensorflow as tf

class TensorflowNode(Node):

    def __init__(self, node_name):
        super().__init__(node_name)


    def load_model(self, frozen_model_path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(frozen_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.session = tf.compat.v1.Session(graph=self.graph)

        self.get_logger().info("Load model completed!")
