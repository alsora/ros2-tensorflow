import rclpy

from ros2_tensorflow.utils import img_conversion as img_utils
from tf_interfaces.srv import ImageDetection as ImageDetectionSrv

IMG_PATH = "/root/ros2-tensorflow/data/dogs.jpg"

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client_test')

    client = node.create_client(ImageDetectionSrv, 'image_detection')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = ImageDetectionSrv.Request()
    req.image = img_utils.jpg_to_image_msg(IMG_PATH)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        node.get_logger().info('Result of classification: %r' % future.result().detections)
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())


    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
