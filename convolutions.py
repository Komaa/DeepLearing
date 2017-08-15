# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
import pickle

import tensorflow as tf

from Util.ScoreMetrics import accuracy

pickle_file = 'Data/notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
num_channels = 1  # grayscale
keep_prob = 0.5

def reformat(dataset, labels):
    # print "before", dataset.shape
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    # print "after", dataset.shape
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 64
patch_size = 5
depth = 16
depth2 = 32
num_hidden = 512
num_hidden2 = 128
beta = 3e-3

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels)) # 16,28,28,1
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))  # 16,10
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))  # Kernel (5,5,1,6)
    layer1_biases = tf.Variable(tf.zeros([depth]))

    layer_2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    # layer2_weights = tf.Variable(tf.truncated_normal(
    #    [patch_size, patch_size, depth, depth], stddev=0.1))    # Kernel (5,5,6,6)
    # layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

    layer3_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth2], stddev=0.1))   # Kernel (5,5,6,10)
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))

    layer_4_biases = tf.Variable(tf.constant(1.0, shape=[depth2]))
    # layer3_weights = tf.Variable(tf.truncated_normal(
    #     [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))  # 784, 64
    # layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer5_weights = tf.Variable(tf.truncated_normal(
        [800, num_hidden], stddev=0.1))  # 784, 120
    layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer6_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, num_hidden2], stddev=0.1))  # 784, 120
    layer6_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

    layer7_weights = tf.Variable(tf.truncated_normal(
        [num_hidden2, num_labels], stddev=0.1))  # 64, 10
    layer7_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    # Model.
    def model_with_dropout(data):
        # pool = tf.nn.max_pool(data, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        print "data", data.shape

        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        print "layer_1", hidden.shape

        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer_2_biases)
        print "layer_2", hidden.shape

        conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + layer3_biases)
        print "layer_3", hidden.shape

        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer_4_biases)
        print "layer_4", hidden.shape

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer5_weights) + layer5_biases)
        hidden = tf.nn.dropout(hidden, keep_prob)
        print "layer_5", hidden.shape

        hidden = tf.nn.relu(tf.matmul(hidden, layer6_weights) + layer6_biases)
        hidden = tf.nn.dropout(hidden, keep_prob)
        print "layer_6", hidden.shape

        return tf.matmul(hidden, layer7_weights) + layer7_biases


    def model_without_dropout(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)

        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer_2_biases)

        conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding='VALID')
        hidden = tf.nn.relu(conv + layer3_biases)

        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(pool + layer_4_biases)

        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer5_weights) + layer5_biases)

        hidden = tf.nn.relu(tf.matmul(hidden, layer6_weights) + layer6_biases)

        return tf.matmul(hidden, layer7_weights) + layer7_biases


    # Training computation.
    logits = model_with_dropout(tf_train_dataset)

    # adding regularizers
    regularizers = (tf.nn.l2_loss(layer5_weights) + tf.nn.l2_loss(layer5_biases) +
                    tf.nn.l2_loss(layer6_weights) + tf.nn.l2_loss(layer6_biases) +
                    tf.nn.l2_loss(layer7_weights) + tf.nn.l2_loss(layer7_biases)
                    )

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + beta * regularizers

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.95, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model_without_dropout(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model_without_dropout(tf_test_dataset))

num_steps = 8001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
