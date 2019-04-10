import argparse
import os
import tensorflow as tf


def str2dir(dir_name):
    if not os.path.isdir(dir_name):
        raise argparse.ArgumentTypeError('{} is not a directory!'.format(dir_name))
    elif os.access(dir_name, os.W_OK) is False:
        raise argparse.ArgumentTypeError('{} is not a writeable directory!'.format(dir_name))
    else:
        return os.path.abspath(os.path.expanduser(dir_name))


def build_accuracy(logits, labels, name_scope='accuracy'):
    """
    Builds a graph node to compute accuracy given 'logits' a probability distribution over the output and 'labels' a
    one-hot vector.
    """
    with tf.name_scope(name_scope):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        return tf.reduce_mean(correct_prediction)
