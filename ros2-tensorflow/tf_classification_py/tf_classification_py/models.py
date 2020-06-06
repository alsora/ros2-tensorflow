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

from ros2_tf_core import models as models_utils
import tensorflow as tf


TENSORFLOW_IMAGENET_DIR = os.path.join(os.path.dirname(tf.__file__), 'models/image/imagenet')

IMAGENET_INCEPTION = models_utils.ModelDescriptor().from_url(
    url='http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz',
    label_path=os.path.join(os.path.join(TENSORFLOW_IMAGENET_DIR, 'inception-2015-12-05'), 'imagenet_2012_challenge_label_map_proto.pbtxt'),  # noqa: E501
    download_directory=TENSORFLOW_IMAGENET_DIR,
    model_filename='classify_image_graph_def.pb',
    save_load_format=models_utils.SaveLoadFormat.FROZEN_MODEL,
    description='TensorFlow inception network for image classification, trained on imagenet')
