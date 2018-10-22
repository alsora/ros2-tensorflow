

from my_interfaces.srv import ClassifyImage

import rclpy
import tensorflow as tf
import os
import numpy as np
from datetime import datetime


tf.flags.DEFINE_string("model_dir", "/root/ros2-tensorflow/models/mnist_model", "Directory containing a trained model, i.e. checkpoints, saved, vocab_words")

FLAGS = tf.flags.FLAGS

session = None

def handle_classify_image_srv(request, response):
    global session

    a = datetime.now()

    x = np.asarray(request.image.data)

    x = x.reshape([-1,784])

    input_x = session.graph.get_operation_by_name("input_x").outputs[0]
    predictions_op = session.graph.get_operation_by_name("output/predictions").outputs[0] 

    feed_dict = {
        input_x: x
    }

    dataset_op = session.graph.get_operation_by_name("dataset_init")

    session.run(dataset_op, feed_dict=feed_dict)

    prediction_array = session.run([predictions_op])

    response.prediction = (np.argmax(prediction_array)).tolist()

    b = datetime.now()
    c = b - a

    print ("handle_classify_image_srv took: ", c)

    return response


def main(args=None):
    global session

    rclpy.init(args=args)

    node = rclpy.create_node('server')

    srv = node.create_service(ClassifyImage, 'classify_image', handle_classify_image_srv)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saved_model_dir = os.path.join(FLAGS.model_dir, "saved")
    tag = [tf.saved_model.tag_constants.SERVING]
    tf.saved_model.loader.load(session, tag, saved_model_dir)

    print ("Model loaded!")

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - Done automatically when node is garbage collected)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
