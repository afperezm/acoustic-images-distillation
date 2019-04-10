import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected],
                        activation_fn=slim.nn_ops.relu,
                        weights_regularizer=slim.regularizers.l2_regularizer(weight_decay),
                        biases_initializer=slim.init_ops.zeros_initializer()):
        with slim.arg_scope([slim.layers.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def build_arg_scope_with_batch_norm(is_training=True,
                                    weight_decay=0.0005,
                                    batch_norm_decay=0.997,
                                    batch_norm_epsilon=0.001,
                                    batch_norm_scale=False):
    """Defines the HearNet arg scope.

    Args:
      is_training:
      weight_decay: The l2 regularization coefficient.
      batch_norm_decay: Batch norm decay for the moving average.
      batch_norm_epsilon: Batch norm added to variance to avoid division by zero.
      batch_norm_scale: Boolean flag to indicated whether to use scaling or not.

    Returns:
      An arg_scope.
    """

    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with slim.arg_scope([slim.layers.conv2d],
                        padding='SAME',
                        activation_fn=slim.nn_ops.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.regularizers.l2_regularizer(weight_decay),
                        biases_initializer=slim.init_ops.zeros_initializer()):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def build_network(inputs,
                  num_classes=14,
                  is_training=True,
                  keep_prob=0.5,
                  scope='hear_net'):
    """
    Builds a three-layer network that operates over a spectrogram.
    """

    with tf.variable_scope(scope, 'hear_net', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv2d, slim.layers.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.layers.conv2d(inputs, 128, [11, 1], [1, 1], scope='conv1')
            net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool1')
            net = slim.layers.conv2d(net, 256, [5, 1], [1, 1], scope='conv2')
            net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool2')
            net = slim.layers.conv2d(net, 256, [3, 1], [1, 1], scope='conv3')
            net = slim.layers.max_pool2d(net, [5, 1], [5, 1], scope='pool3')
            net = slim.layers.conv2d(net, 1024, [4, 1], padding='VALID', scope='conv4')
            net = slim.layers.conv2d(net, 1024, 1, scope='conv5')

    # Convert end_points_collection into a end_point dictionary
    end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

    return net, end_points
