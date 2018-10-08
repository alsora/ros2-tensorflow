from my_interfaces.srv import ClassifyImage
from sensor_msgs.msg import Image

import rclpy
import cv2

from time import sleep


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client')

    cli = node.create_client(ClassifyImage, 'classify_image')
    while not cli.wait_for_service(timeout_sec=1.0):
        print('service not available, waiting again...')

    img = cv2.imread("/root/ros2-tensorflow/fig.png", cv2.IMREAD_GRAYSCALE)

    img_msg = Image()

    img_msg.height = img.shape[0]
    img_msg.width = img.shape[1]
    img_msg.encoding = "mono8"
    img_msg.data = img.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    img_msg.header.frame_id = "world"

    req = ClassifyImage.Request()

    req.image = img_msg

    while (1):

        future = cli.call_async(req)
        rclpy.spin_until_future_complete(node, future)
        if future.result() is not None:
            node.get_logger().info('Result of classification: %d' % future.result().prediction)
        else:
            node.get_logger().error('Exception while calling service: %r' % future.exception())

        sleep(0.25)


    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
