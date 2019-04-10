import tensorflow as tf
import tensorflow.contrib.slim as slim


def dualcamnet_v2(inputs,
                  keep_prob=0.5,
                  is_training=True,
                  num_classes=None,
                  num_frames=12,
                  num_channels=12,
                  spatial_squeeze=False, scope='DualCamNet'):
    """
    Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
    """

    with tf.variable_scope(scope, 'DualCamNet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv3d, slim.layers.conv2d, slim.layers.max_pool2d],
                            outputs_collections=[end_points_collection]):
            # ----------- 1st layer group ---------------
            net = tf.reshape(inputs, shape=(-1, num_frames, 36, 48, num_channels))
            net = slim.conv3d(net, num_channels, [7, 1, 1], scope='conv1', padding='SAME')
            net = tf.reshape(net, shape=(-1, 36, 48, num_channels))
            # ----------- 2nd layer group ---------------
            net = slim.conv2d(net, 32, [5, 5], scope='conv2', padding='SAME')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            # ----------- 3rd layer group ---------------
            net = slim.conv2d(net, 64, [5, 5], scope='conv3', padding='SAME')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            if num_classes is None:
                # ----------- 4th layer group ---------------
                net = slim.conv2d(net, 1024, [9, 12], scope='full1', padding='VALID')
            else:
                # ----------- 4th layer group ---------------
                net = slim.conv2d(net, 1024, [9, 12], scope='full1', padding='VALID')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='drop1')
                # ----------- 5th layer group ---------------
                net = slim.conv2d(net, 1000, 1, scope='full2')
                # ----------- 6th layer group ---------------
                net = slim.conv2d(net, num_classes, 1, scope='full3')

            # Convert end_points_collection into a end_point dictionary
            end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='full3/squeezed')
                end_points[sc.name + '/full3'] = net

            return net, end_points
