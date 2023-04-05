# Copyright 2020 Alberto Soragna. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import rclpy
from ros2_tf_core import img_conversion as img_utils
from ros2_tf_core.qos import qos_profile_vision_info
import tensorflow as tf
from tf_interfaces.srv import ImageDetection as ImageDetectionSrv
from vision_msgs.msg import VisionInfo as VisionInfoMsg


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client_test')

    client = node.create_client(ImageDetectionSrv, 'image_detection')

    def vision_info_callback(vision_info_msg):
        node.get_logger().info(f'received vision info {repr(vision_info_msg)}')

    node.create_subscription(
        VisionInfoMsg, 'vision_info', vision_info_callback, qos_profile=qos_profile_vision_info)

    img_path = tf.keras.utils.get_file(
        'BearAndDog.jpg',
        'http://farm9.staticflickr.com/8036/7946162344_ee2e1a814e_z.jpg')

    while not client.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = ImageDetectionSrv.Request()
    req.image = img_utils.jpg_to_image_msg(img_path)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    if future.result() is not None:
        detections = future.result().detections.detections
        for det in detections:
            det_result = det.results[0]
            node.get_logger().info(
                f'Detected object {det_result.hypothesis.class_id} with score {det_result.hypothesis.score}')
    else:
        node.get_logger().error(f'Exception while calling service: {future.exception()}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
