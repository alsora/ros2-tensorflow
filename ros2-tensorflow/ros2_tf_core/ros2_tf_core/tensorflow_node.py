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

from rclpy.node import Node

from ros2_tf_core.qos import qos_profile_vision_info
from vision_msgs.msg import VisionInfo


class TensorflowNode(Node):

    def __init__(self, node_name):
        super().__init__(node_name)

    def publish_vision_info(self, tf_model):

        # Publish vision info message (published only once with TRANSIENT LOCAL durability)
        vision_info_msg = VisionInfo()
        vision_info_msg.method = tf_model.description
        vision_info_msg.database_location = tf_model.compute_label_path()

        vision_info_pub = self.create_publisher(
            VisionInfo, 'vision_info', qos_profile=qos_profile_vision_info)
        vision_info_pub.publish(vision_info_msg)
