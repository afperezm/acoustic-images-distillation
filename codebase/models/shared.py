import tensorflow as tf
import tensorflow.contrib.slim as slim


def shared_net(inputs, num_classes=None, is_training=True, keep_prob=0.5, spatial_squeeze=True, scope='shared_net'):
    """
    Builds a three-layer fully-connected modality agnostic network.
    """

    with tf.variable_scope(scope, [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv2d], padding='VALID', outputs_collections=[end_points_collection],
                            weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01),
                            biases_initializer=tf.constant_initializer(0.0)):
            # Use convolution2d instead of fully_connected layers
            net = slim.layers.conv2d(inputs, 1000, 1, scope='fc1')
            net = slim.layers.conv2d(net, num_classes, 1, activation_fn=None, normalizer_fn=None, scope='fc2')
            # Convert end_points_collection into a end_point dictionary
            end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                # Apply spatial squeezing
                net = tf.squeeze(net, [1, 2], name='fc2/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/fc2'] = net

    return net, end_points
