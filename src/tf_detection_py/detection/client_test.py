from my_interfaces.srv import ImageDetection
from sensor_msgs.msg import Image

#from cv_bridge import CvBridge

import rclpy
import cv2

from time import sleep


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client')

    cli = node.create_client(ImageDetection, 'image_detection')
    while not cli.wait_for_service(timeout_sec=1.0):
        print('service not available, waiting again...')


    #br = CvBridge()
    #dtype, n_channels = br.encoding_as_cvtype2('8UC3')

    IMG_PATH = "/root/ros2-tensorflow/data/dogs.jpg"
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

    #img_msg = br.cv2_to_imgmsg(img) 

    img_msg = Image()
    img_msg.height = img.shape[0]
    img_msg.width = img.shape[1]
    img_msg.encoding = "rgb8"
    img_msg.data = img.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height
    img_msg.header.frame_id = "world"

    req = ImageDetection.Request()

    req.image = img_msg


    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Result of classification: %r' % future.result().detections)
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())




    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
