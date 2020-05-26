
from rclpy.node import Node


class TensorflowNode(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
