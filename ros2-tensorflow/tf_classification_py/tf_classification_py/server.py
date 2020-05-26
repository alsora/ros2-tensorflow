import rclpy

from tf_classification_py.classification_node import ClassificationNode


def main(args=None):

    rclpy.init(args=args)

    node = ClassificationNode('classification_server')
    node.create_classification_server('image_classification')
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
