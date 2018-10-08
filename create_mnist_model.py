from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':


    #define network
    
    input_x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    input_y = tf.placeholder_with_default(tf.zeros([1, 10], tf.float32), [None, 10], name="input_y")
    dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    learning_rate = 1e-4

    dataset = tf.data.Dataset.from_tensor_slices((input_x, input_y))
    dataset  = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(50)
    dataset = dataset.repeat()

    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

    dataset_init_op = iterator.make_initializer(dataset, name='dataset_init')

    x, y = iterator.get_next()

    x_image = tf.reshape(x, [-1,28,28,1])

    with tf.variable_scope("conv-maxpool-1"):

        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.variable_scope("conv-maxpool-2"):
    
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)
    
    with tf.variable_scope("fully-connected"): 

        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.variable_scope("dropout"):

        h_fc1_drop = tf.nn.dropout(h_fc1, dropout_keep_prob)

    with tf.variable_scope("output"):        

        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        y_conv=tf.nn.softmax(logits, name="predictions")

    with tf.variable_scope("loss"):

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]), name="loss")

        opt = tf.train.AdamOptimizer(learning_rate)

        grads_and_vars = opt.compute_gradients(cross_entropy)

        optimizer = opt.apply_gradients(grads_and_vars, global_step=global_step, name="optimizer")

    with tf.variable_scope("accuracy"):

        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


    #train network

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    feed_dict = {
        input_x: mnist.train.images,
        input_y: mnist.train.labels
    }

    sess.run(dataset_init_op, feed_dict=feed_dict)

    for e in range(2):

        train_batches_per_epoch = (len(mnist.train.images) - 1) // 50 + 1

        for i in range(train_batches_per_epoch):

            _, _, acc = sess.run([optimizer, global_step, accuracy])

            if i%100 == 0:
                print("step %d, training accuracy %.3f"%(i, acc))

    #save model

    builder = tf.saved_model.builder.SavedModelBuilder("models/mnist_model/saved")
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        clear_devices=True)

    builder.save()