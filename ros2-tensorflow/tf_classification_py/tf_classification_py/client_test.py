# Copyright 2020. Alberto Soragna. All Rights Reserved.
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

from ros2_tensorflow.node.qos import qos_profile_vision_info
from ros2_tensorflow.utils import img_conversion as img_utils

from tf_interfaces.srv import ImageClassification as ImageClassificationSrv
from vision_msgs.msg import VisionInfo as VisionInfoMsg

IMG_PATH = '/root/ros2-tensorflow/data/dog.jpg'


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('client_test')

    client = node.create_client(ImageClassificationSrv, 'image_classification')

    vision_info_callback = lambda info_msg : node.get_logger().info('received vision info %r' % info_msg)
    info_subscription = node.create_subscription(VisionInfoMsg, 'vision_info', vision_info_callback, qos_profile=qos_profile_vision_info)

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
