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

import numpy as np
import tensorflow as tf

from ros2_tensorflow.utils import models as models_utils


class FrozenDetectionModel():

    def __init__(self, tf_model):

        if tf_model.save_load_format != models_utils.SaveLoadFormat.FROZEN_MODEL:
            raise ValueError('Creating a FrozenDetectionModel from a TensorflowModel with invalid format')

        # Load the model
        model_path = tf_model.compute_model_path()
        self.graph, self.session = models_utils.load_frozen_model(model_path)

        # Define input tensor
        self.input_image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Define output tensors
        self.output_tensor_dict = {}
        self.output_tensor_dict['detection_boxes'] = self.graph.get_tensor_by_name('detection_boxes:0')
        self.output_tensor_dict['detection_classes'] = self.graph.get_tensor_by_name('detection_classes:0')
        self.output_tensor_dict['detection_scores'] = self.graph.get_tensor_by_name('detection_scores:0')
        self.output_tensor_dict['num_detections'] = self.graph.get_tensor_by_name('num_detections:0')

    def inference(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Perform the inference
        output_dict = self.session.run(
            self.output_tensor_dict,
            feed_dict={self.input_image_tensor: image_np_expanded})

        # Reshape the tensors:
        # - squeeze to remove the batch dimension (since here we fed a single image)
        # - keep only the first num_detections elements
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections] for key, value in output_dict.items()}

        # Convert classes from float to int (values are already numpy arrays)
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.uint8)

        return output_dict


class SavedDetectionModel():

    def __init__(self, tf_model):

        if tf_model.save_load_format != models_utils.SaveLoadFormat.SAVED_MODEL:
            raise ValueError('Creating a FrozenDetectionModel from a TensorflowModel with invalid format')

        # Load model
        model_path = tf_model.compute_model_path()
        self.model = models_utils.load_saved_model(model_path)

    def inference(self, image_np):
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        input_tensor = tf.convert_to_tensor(image_np_expanded)

        # Perform the inference
        output_dict = self.model(input_tensor)

        # Reshape the tensors:
        # - squeeze to remove the batch dimension (since here we fed a single image)
        # - keep only the first num_detections elements
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key: value[0, :num_detections] for key, value in output_dict.items()}

        # Convert the tensors into numpy arrays
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/ops.py#L1047
        output_dict['detection_classes'] = output_dict['detection_classes'].numpy().astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'].numpy()
        output_dict['detection_scores'] = output_dict['detection_scores'].numpy()

        return output_dict


def create(tf_model):
    # Create the model according to the specified format
    return {
        models_utils.SaveLoadFormat.FROZEN_MODEL: lambda model: FrozenDetectionModel(tf_model),
        models_utils.SaveLoadFormat.SAVED_MODEL: lambda model: SavedDetectionModel(tf_model)
    }[tf_model.save_load_format](tf_model)
