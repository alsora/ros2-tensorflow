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

import os

import object_detection
from ros2_tf_core import models as models_utils


TENSORFLOW_OBJECT_DETECTION_DIR = os.path.dirname(object_detection.__file__)
COCO_LABELS = os.path.join(TENSORFLOW_OBJECT_DETECTION_DIR, 'data/mscoco_label_map.pbtxt')

COCO_FASTER_RCNN = models_utils.ModelDescriptor().from_url(
    url='http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz',  # noqa: E501
    label_path=COCO_LABELS,
    download_directory=TENSORFLOW_OBJECT_DETECTION_DIR,
    model_filename='saved_model',
    save_load_format=models_utils.SaveLoadFormat.SAVED_MODEL,
    description='TensorFlow Faster RCNN Inception network for object detection. '
        'Trained on COCO dataset. Produces boxes.')

COCO_MOBILENET = models_utils.ModelDescriptor().from_url(
    url='http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz',  # noqa: E501
    label_path=COCO_LABELS,
    download_directory=TENSORFLOW_OBJECT_DETECTION_DIR,
    model_filename='saved_model',
    save_load_format=models_utils.SaveLoadFormat.SAVED_MODEL,
    description='TensorFlow Mobilenet network for object detection. '
        'Trained on COCO dataset. Produces boxes.')

COCO_MASK_RCNN = models_utils.ModelDescriptor().from_url(
    url='http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz',  # noqa: E501
    label_path=COCO_LABELS,
    download_directory=TENSORFLOW_OBJECT_DETECTION_DIR,
    model_filename='saved_model',
    save_load_format=models_utils.SaveLoadFormat.SAVED_MODEL,
    description='TensorFlow RCNN Inception network for object detection. '
        'Trained on COCO dataset. Produces masks.')
