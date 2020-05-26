import rclpy

from ros2_tensorflow.utils import img_conversion as img_utils
from tf_interfaces.srv import ImageClassification as ImageClassificationSrv

IMG_PATH = '/root/ros2-tensorflow/data/dog.jpg'


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client_test')

    client = node.create_client(ImageClassificationSrv, 'image_classification')
    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = ImageClassificationSrv.Request()
    req.image = img_utils.jpg_to_image_msg(IMG_PATH)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        classification = future.result().classification
        node.get_logger().info('Result of classification:\n' + str(classification.results))
    else:
        node.get_logger().error('Exception while calling service: %r' % future.exception())

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
