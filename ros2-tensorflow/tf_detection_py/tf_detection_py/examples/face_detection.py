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
from ros2_tf_core import models as models_utils
from tf_detection_py.detection_node import DetectionNode
from tf_detection_py.models import TENSORFLOW_OBJECT_DETECTION_DIR


def main(args=None):

    # Download a model and labels for face detections
    # Courtesy of https://github.com/yeephycho/tensorflow-face-detection
    model_path = models_utils.maybe_download_and_extract(
        'https://docs.google.com/uc?export=download&id=0B5ttP5kO_loUdWZWZVVrN2VmWFk',
        TENSORFLOW_OBJECT_DETECTION_DIR,
        'frozen_inference_graph_face.pb',
        extract=False)
    label_path = models_utils.maybe_download_and_extract(
        'https://raw.githubusercontent.com/yeephycho/tensorflow-face-detection/master/protos/face_label_map.pbtxt',  # noqa: E501
        TENSORFLOW_OBJECT_DETECTION_DIR,
        'face_label_map.pbtxt',
        extract=False)

    # Create a descriptor for the just downloaded network
    SSD_FACE_DETECTION = models_utils.ModelDescriptor().from_path(
        model_path=model_path,
        label_path=label_path,
        save_load_format=models_utils.SaveLoadFormat.FROZEN_MODEL,
        description='TensorFlow SSD Face Detection, trained on WIDERFACE dataset. Produces boxes')

    rclpy.init(args=args)

    node = DetectionNode(SSD_FACE_DETECTION, 'detection_server')
    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
