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

from tf_classification_py.classification_node import ClassificationNode
from tf_classification_py.models import IMAGENET_INCEPTION


def main(args=None):

    rclpy.init(args=args)

    node = ClassificationNode(IMAGENET_INCEPTION, 'classification_server')
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
