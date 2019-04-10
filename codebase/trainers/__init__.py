import tensorflow as tf


def build_accuracy(logits, labels, name_scope='accuracy'):
    """
    Builds a graph node to compute accuracy given 'logits' a probability distribution over the output and 'labels' a
    one-hot vector.
    """
    with tf.name_scope(name_scope):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)
