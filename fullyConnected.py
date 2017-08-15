import pickle
import numpy as np
import tensorflow as tf

pickle_file = 'Data/notMNIST.pickle'
image_size = 28
num_labels = 10


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def accuracy(predicts, labels):
    return (100.0 * np.sum(np.argmax(predicts, 1) == np.argmax(labels, 1))
            / predicts.shape[0])


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

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
batch_size = 256
num_hidden_nodes1 = 1024
num_hidden_nodes2 = 500
num_hidden_nodes3 = 50
beta = 1e-3
num_steps = 30001
keep_prob = 0.5

graph = tf.Graph()
with graph.as_default():

    # This for stochastic gradient descent
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    global_step = tf.Variable(0)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_hidden_nodes1],
                                               stddev=np.sqrt(2.0 / (image_size * image_size))))
    biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))

    weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2],
                                               stddev=np.sqrt(2.0 / num_hidden_nodes1)))
    biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))

    weights3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3],
                                               stddev=np.sqrt(2.0 / num_hidden_nodes2)))
    biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))

    weights = tf.Variable(tf.truncated_normal([num_hidden_nodes3, num_labels],
                                              stddev=np.sqrt(2.0 / num_hidden_nodes3)))
    biases = tf.Variable(tf.zeros([num_labels]))


    # Training computation.
    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights2) + biases2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)

    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights3) + biases3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    logits = tf.matmul(layer_3, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)) + \
           beta * (tf.nn.l2_loss(weights)+tf.nn.l2_loss(weights1)+tf.nn.l2_loss(weights2))

    # Optimizer.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)

    valid_layer_1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
    valid_layer_2 = tf.nn.relu(tf.matmul(valid_layer_1, weights2) + biases2)
    valid_layer_3 = tf.nn.relu(tf.matmul(valid_layer_2, weights3) + biases3)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_layer_3, weights) + biases)

    test_layer_1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
    test_layer_2 = tf.nn.relu(tf.matmul(test_layer_1, weights2) + biases2)
    test_layer_3 = tf.nn.relu(tf.matmul(test_layer_2, weights3) + biases3)
    test_prediction = tf.nn.softmax(tf.matmul(test_layer_3, weights) + biases)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
