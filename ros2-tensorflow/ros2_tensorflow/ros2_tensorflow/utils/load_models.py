import tensorflow as tf


def load_frozen_model(frozen_model_path):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(frozen_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    session = tf.compat.v1.Session(graph=graph)
    return graph, session
