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

import cv2

import numpy as np

from sensor_msgs.msg import Image as ImageMsg


def image_msg_to_image_np(image_msg):
    n_channels = 3
    dtype = 'uint8'
    img_buf = np.asarray(image_msg.data, dtype=dtype)

    image_np = np.ndarray(shape=(image_msg.height, image_msg.width, n_channels),
                          dtype=dtype, buffer=img_buf)

    return image_np


def image_np_to_image_msg(image_np):
    image_msg = ImageMsg()

    image_msg.height = image_np.shape[0]
    image_msg.width = image_np.shape[1]
    image_msg.encoding = 'bgr8'
    image_msg.data = image_np.tostring()
    image_msg.step = len(image_msg.data) // image_msg.height
    image_msg.header.frame_id = 'map'

    return image_msg


def jpg_to_image_msg(img_path):

    # br = CvBridge()
    # dtype, n_channels = br.encoding_as_cvtype2('8UC3')
    # image_msg = br.cv2_to_imgmsg(img)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    image_msg = ImageMsg()
    image_msg.height = img.shape[0]
    image_msg.width = img.shape[1]
    image_msg.encoding = 'rgb8'
    image_msg.data = img.tostring()
    image_msg.step = len(image_msg.data) // image_msg.height

    return image_msg
