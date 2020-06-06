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

import numpy as np

from object_detection.utils import ops as utils_ops
from ros2_tf_core import models as models_utils
import tensorflow as tf


class DetectionFrozenModel():

    def __init__(self, tf_model):

        if tf_model.save_load_format != models_utils.SaveLoadFormat.FROZEN_MODEL:
            raise ValueError(
                'Creating a DetectionFrozenModel from a ModelDescriptor with invalid format')

        # Load the model
        model_path = tf_model.compute_model_path()
        self.graph, self.session = models_utils.load_frozen_model(model_path)

        # Define input tensor
        self.input_image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Define output tensors
        self.output_tensors = {}
        self.output_tensors['detection_boxes'] = self.graph.get_tensor_by_name(
            'detection_boxes:0')
        self.output_tensors['detection_classes'] = self.graph.get_tensor_by_name(
            'detection_classes:0')
        self.output_tensors['detection_scores'] = self.graph.get_tensor_by_name(
            'detection_scores:0')
        self.output_tensors['num_detections'] = self.graph.get_tensor_by_name(
            'num_detections:0')

        # The following output tensors are optional
        ops = self.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        if 'detection_masks:0' in all_tensor_names:
            self.output_tensors['detection_masks'] = self.graph.get_tensor_by_name(
                'detection_masks:0')

    def inference(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Perform the inference
        output = self.session.run(
            self.output_tensors,
            feed_dict={self.input_image_tensor: image_np_expanded})

        # Reshape the tensors:
        # - squeeze to remove the batch dimension (since here we fed a single image)
        # - keep only the first num_detections elements
        num_detections = int(output.pop('num_detections'))
        output = {key: value[0, :num_detections] for key, value in output.items()}

        # output_dict already contains numpy arrays
        # Convert classes from float to int
        output['detection_classes'] = output['detection_classes'].astype(np.uint8)

        # The following output tensors are optional
        if 'detection_masks' in output:
            reframe_detection_masks(output, image_np)
            output['detection_masks'] = output['detection_masks'].numpy()

        return output


class DetectionSavedModel():

    def __init__(self, tf_model):

        if tf_model.save_load_format != models_utils.SaveLoadFormat.SAVED_MODEL:
            raise ValueError(
                'Creating a DetectionSavedModel from a ModelDescriptor with invalid format')

        # Load model
        model_path = tf_model.compute_model_path()
        self.model = models_utils.load_saved_model(model_path)

    def inference(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Convert to tensor
        input_tensor = tf.convert_to_tensor(image_np_expanded)

        # Perform the inference
        output = self.model(input_tensor)

        # Reshape the tensors:
        # - squeeze to remove the batch dimension (since here we fed a single image)
        # - keep only the first num_detections elements
        num_detections = int(output.pop('num_detections'))
        output = {key: value[0, :num_detections] for key, value in output.items()}

        # Convert the tensors into numpy arrays
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py#L1047
        output['detection_classes'] = output['detection_classes'].numpy().astype(np.int64)
        output['detection_boxes'] = output['detection_boxes'].numpy()
        output['detection_scores'] = output['detection_scores'].numpy()

        # The following output tensors are optional
        if 'detection_masks' in output:
            reframe_detection_masks(output, image_np)
            output['detection_masks'] = output['detection_masks'].numpy()

        return output


def create(tf_model):
    # Create the model according to the specified format
    return {
        models_utils.SaveLoadFormat.FROZEN_MODEL: lambda model: DetectionFrozenModel(tf_model),
        models_utils.SaveLoadFormat.SAVED_MODEL: lambda model: DetectionSavedModel(tf_model)
    }[tf_model.save_load_format](tf_model)


def reframe_detection_masks(output_dict, image_np):
    # Reframe the the bbox mask to the image size.
    # Original masks are represented in bounding box coordinates
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image_np.shape[0], image_np.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
    output_dict['detection_masks'] = detection_masks_reframed
