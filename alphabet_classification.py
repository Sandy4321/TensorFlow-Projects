from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'notMNIST.pickle'

try:
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
except Exception as e:
    print('Unable to process', pickle_file, ':', e)
    raise


image_size = 28
num_labels = 10
# num_channels = 1


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


batch_size = 228

graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = {
        'l1': tf.Variable(tf.truncated_normal([image_size * image_size, 1024], stddev=0.1)),
        'l2': tf.Variable(tf.truncated_normal([1024, 600], stddev=0.1)),
        'l3': tf.Variable(tf.truncated_normal([600, 300], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([300, num_labels], stddev=0.1))
    }

    biases = {
        'l1': tf.Variable(tf.zeros([1024])),
        'l2': tf.Variable(tf.zeros([600])),
        'l3': tf.Variable(tf.zeros([300])),
        'out': tf.Variable(tf.zeros([num_labels]))
        # 'l1': tf.Variable(tf.constant(0.1, shape=[1024])),
        # 'out': tf.Variable(tf.constant(0.1, shape=[10]))
    }
    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.

    def multilayered_network(x, weights, biases):

        hidden_layer = tf.nn.relu(tf.matmul(x, weights['l1'])) + biases['l1']

        hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer, weights['l2'])) + biases['l2']

        hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, weights['l3'])) +biases['l3']

        logits = tf.matmul(hidden_layer_3, weights['out']) + biases['out']

        return logits

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(multilayered_network(tf_train_dataset, weights, biases))

    beta = 0.001

    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(train_prediction, tf_train_labels)
            + beta*tf.nn.l2_loss(weights['l1']) +
            beta*tf.nn.l2_loss(biases['l1']) +
            beta*tf.nn.l2_loss(weights['out']) +
            beta*tf.nn.l2_loss(weights['out']))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    global_step = tf.Variable(0)  # count the number of steps taken.
    learning_rate = tf.train.exponential_decay(0.5, global_step, 500, 0.96)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    valid_prediction = tf.nn.softmax(multilayered_network(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(multilayered_network(tf_test_dataset, weights, biases))

num_steps = 5001


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the
    # biases.
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Validation accuracy: %.1f%%" % accuracy(
                valid_prediction.eval(), valid_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            # print('Learning rate: %.1f%%' % learning_rate)

    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
