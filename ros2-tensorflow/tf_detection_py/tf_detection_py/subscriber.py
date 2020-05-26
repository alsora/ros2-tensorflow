import rclpy

from tf_detection_py.detection_node import DetectionNode


def main(args=None):

    rclpy.init(args=args)

    node = DetectionNode('detection_subscriber')
    node.create_detection_subscription('image')
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
