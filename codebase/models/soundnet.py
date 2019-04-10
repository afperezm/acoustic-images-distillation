import tensorflow as tf
import tensorflow.contrib.slim as slim
import torchfile


def soundnet_arg_scope(is_training=True,
                       weight_decay=0.0001):
    """Defines the SoundNet arg scope.

    Args:
      is_training: Boolean flag indicating whether we are in training or not.
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """

    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.regularizers.l2_regularizer(weight_decay),
            padding='VALID',
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01),
            biases_initializer=tf.constant_initializer(0.0)):
        with slim.arg_scope([slim.batch_norm],
                            scale=True,
                            activation_fn=slim.nn_ops.relu,
                            is_training=is_training) as arg_sc:
            return arg_sc


def soundnet5(inputs,
              num_classes=None,
              spatial_squeeze=False,
              scope='SoundNet'):
    """
    Builds a SoundNet 5-Layers network.
    """

    with tf.variable_scope(scope, 'SoundNet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv2d, slim.batch_norm, slim.layers.max_pool2d],
                            outputs_collections=[end_points_collection]):

            # ----------- 1st layer group ---------------
            net = tf.pad(inputs, [[0, 0], [32, 32], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 32, [64, 1], [2, 1], scope='conv1')
            net = slim.batch_norm(net, scope='conv1/norm')
            net = slim.max_pool2d(net, [8, 1], [8, 1], scope='pool1')
            # ----------- 2nd layer group ---------------
            net = tf.pad(net, [[0, 0], [16, 16], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 64, [32, 1], [2, 1], scope='conv2')
            net = slim.batch_norm(net, scope='conv2/norm')
            net = slim.max_pool2d(net, [8, 1], [8, 1], scope='pool2')
            # ----------- 3rd layer group ---------------
            net = tf.pad(net, [[0, 0], [8, 8], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 128, [16, 1], [2, 1], scope='conv3')
            net = slim.batch_norm(net, scope='conv3/norm')
            net = slim.max_pool2d(net, [8, 1], [8, 1], scope='pool3')
            # ----------- 4th layer group ---------------
            net = tf.pad(net, [[0, 0], [4, 4], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 256, [8, 1], [2, 1], scope='conv4')
            net = slim.batch_norm(net, scope='conv4/norm')
            # ----------- 5th layer group ---------------
            net = tf.pad(net, [[0, 0], [4, 4], [0, 0], [0, 0]], 'CONSTANT')
            if num_classes is None:
                conv5a = slim.conv2d(net, 1000, [16, 1], [12, 1], scope='conv5a')
                conv5b = slim.conv2d(net, 401, [16, 1], [12, 1], scope='conv5b')
                net = (conv5a, conv5b)
            else:
                net = slim.conv2d(net, 1024, [16, 1], [12, 1], scope='conv5')
                net = slim.batch_norm(net, scope='conv5/norm')
                net = slim.conv2d(net, 1024, 1, scope='conv6', activation_fn=slim.nn_ops.relu)

            # Convert end_points_collection into a end_point dictionary
            end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

            if num_classes is not None and spatial_squeeze:
                # Apply spatial squeezing
                net = tf.squeeze(net, [1, 2], name='conv6/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/conv6'] = net

            return net, end_points


soundnet5.default_size = [22050 * 5, 1, 1]


def soundnet8(inputs,
              num_classes=None,
              spatial_squeeze=True,
              scope='SoundNet'):
    """
    Builds a SoundNet 8-Layers network.
    """

    with tf.variable_scope(scope, 'SoundNet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        # Collect outputs for convolution2d and max_pool2d
        with slim.arg_scope([slim.layers.conv2d, slim.batch_norm, slim.layers.max_pool2d],
                            outputs_collections=[end_points_collection]):

            # ----------- 1st layer group ---------------
            net = tf.pad(inputs, [[0, 0], [32, 32], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 16, [64, 1], [2, 1], scope='conv1')
            net = slim.batch_norm(net, scope='conv1/norm')
            net = slim.max_pool2d(net, [8, 1], [8, 1], scope='pool1')
            # ----------- 2nd layer group ---------------
            net = tf.pad(net, [[0, 0], [16, 16], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 32, [32, 1], [2, 1], scope='conv2')
            net = slim.batch_norm(net, scope='conv2/norm')
            net = slim.max_pool2d(net, [8, 1], [8, 1], scope='pool2')
            # ----------- 3rd layer group ---------------
            net = tf.pad(net, [[0, 0], [8, 8], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 64, [16, 1], [2, 1], scope='conv3')
            net = slim.batch_norm(net, scope='conv3/norm')
            # ----------- 4th layer group ---------------
            net = tf.pad(net, [[0, 0], [4, 4], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 128, [8, 1], [2, 1], scope='conv4')
            net = slim.batch_norm(net, scope='conv4/norm')
            # ----------- 5th layer group ---------------
            net = tf.pad(net, [[0, 0], [2, 2], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 256, [4, 1], [2, 1], scope='conv5')
            net = slim.batch_norm(net, scope='conv5/norm')
            net = slim.max_pool2d(net, [4, 1], [4, 1], scope='pool5')
            # ----------- 6th layer group ---------------
            net = tf.pad(net, [[0, 0], [2, 2], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 512, [4, 1], [2, 1], scope='conv6')
            net = slim.batch_norm(net, scope='conv6/norm')
            # ----------- 7th layer group ---------------
            net = tf.pad(net, [[0, 0], [2, 2], [0, 0], [0, 0]], 'CONSTANT')
            net = slim.conv2d(net, 1024, [4, 1], [2, 1], scope='conv7')
            net = slim.batch_norm(net, scope='conv7/norm')
            # ----------- 8th layer group ---------------
            if num_classes is None:
                conv8a = slim.conv2d(net, 1000, [8, 1], [2, 1], scope='conv8a')
                conv8b = slim.conv2d(net, 401, [8, 1], [2, 1], scope='conv8b')
                net = (conv8a, conv8b)
            else:
                net = slim.conv2d(net, 1024, [8, 1], [2, 1], scope='conv8')
                net = slim.batch_norm(net, scope='conv8/norm')
                net = slim.conv2d(net, 1024, 1, scope='conv9', activation_fn=slim.nn_ops.relu)

            # Convert end_points_collection into a end_point dictionary
            end_points = slim.layers.utils.convert_collection_to_dict(end_points_collection)

            if num_classes is not None and spatial_squeeze:
                # Apply spatial squeezing
                net = tf.squeeze(net, [1, 2], name='conv9/squeezed')
                # Add squeezed fc1 to the collection of end points
                end_points[sc.name + '/conv9'] = net

            return net, end_points


soundnet8.default_size = [22050 * 5, 1, 1]


def soundnet5_model_params(model_filename=None, num_classes=None, scope='SoundNet'):
    """
    Load model parameters from Torch file.
    """

    def retrieve_if_not_none(local_net, layer_num, var_name): return None if local_net is None else \
        local_net['modules'][layer_num][var_name]

    def transpose_if_not_none(x): return None if x is None else x.transpose((2, 3, 1, 0))

    # Load network
    net = None if model_filename is None else torchfile.load(model_filename)
    sub_net = None if net is None else net['modules'][15]

    # Extract weights and biases
    net_params = dict()

    net_params[scope + '/conv1/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 0, 'weight'))
    net_params[scope + '/conv1/biases'] = retrieve_if_not_none(net, 0, 'bias')

    net_params[scope + '/conv1/norm/gamma'] = retrieve_if_not_none(net, 1, 'weight')
    net_params[scope + '/conv1/norm/beta'] = retrieve_if_not_none(net, 1, 'bias')
    net_params[scope + '/conv1/norm/moving_mean'] = retrieve_if_not_none(net, 1, 'running_mean')
    net_params[scope + '/conv1/norm/moving_variance'] = retrieve_if_not_none(net, 1, 'running_var')

    net_params[scope + '/conv2/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 4, 'weight'))
    net_params[scope + '/conv2/biases'] = retrieve_if_not_none(net, 4, 'bias')

    net_params[scope + '/conv2/norm/gamma'] = retrieve_if_not_none(net, 5, 'weight')
    net_params[scope + '/conv2/norm/beta'] = retrieve_if_not_none(net, 5, 'bias')
    net_params[scope + '/conv2/norm/moving_mean'] = retrieve_if_not_none(net, 5, 'running_mean')
    net_params[scope + '/conv2/norm/moving_variance'] = retrieve_if_not_none(net, 5, 'running_var')

    net_params[scope + '/conv3/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 8, 'weight'))
    net_params[scope + '/conv3/biases'] = retrieve_if_not_none(net, 8, 'bias')

    net_params[scope + '/conv3/norm/gamma'] = retrieve_if_not_none(net, 9, 'weight')
    net_params[scope + '/conv3/norm/beta'] = retrieve_if_not_none(net, 9, 'bias')
    net_params[scope + '/conv3/norm/moving_mean'] = retrieve_if_not_none(net, 9, 'running_mean')
    net_params[scope + '/conv3/norm/moving_variance'] = retrieve_if_not_none(net, 9, 'running_var')

    net_params[scope + '/conv4/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 12, 'weight'))
    net_params[scope + '/conv4/biases'] = retrieve_if_not_none(net, 12, 'bias')

    net_params[scope + '/conv4/norm/gamma'] = retrieve_if_not_none(net, 13, 'weight')
    net_params[scope + '/conv4/norm/beta'] = retrieve_if_not_none(net, 13, 'bias')
    net_params[scope + '/conv4/norm/moving_mean'] = retrieve_if_not_none(net, 13, 'running_mean')
    net_params[scope + '/conv4/norm/moving_variance'] = retrieve_if_not_none(net, 13, 'running_var')

    if num_classes is None:
        net_params[scope + '/conv5a/weights'] = transpose_if_not_none(retrieve_if_not_none(sub_net, 0, 'weight'))
        net_params[scope + '/conv5a/biases'] = retrieve_if_not_none(sub_net, 0, 'bias')

        net_params[scope + '/conv5b/weights'] = transpose_if_not_none(retrieve_if_not_none(sub_net, 1, 'weight'))
        net_params[scope + '/conv5b/biases'] = retrieve_if_not_none(sub_net, 1, 'bias')

    return net_params


def soundnet8_model_params(model_filename=None, num_classes=None, scope='SoundNet'):
    """
    Load model parameters from Torch file.
    """

    def retrieve_if_not_none(local_net, layer_num, var_name): return None if local_net is None else \
        local_net['modules'][layer_num][var_name]

    def transpose_if_not_none(x): return None if x is None else x.transpose((2, 3, 1, 0))

    # Load network
    net = None if model_filename is None else torchfile.load(model_filename)
    sub_net = None if net is None else net['modules'][24]

    # Extract weights and biases
    net_params = dict()

    net_params[scope + '/conv1/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 0, 'weight'))
    net_params[scope + '/conv1/biases'] = retrieve_if_not_none(net, 0, 'bias')

    net_params[scope + '/conv1/norm/gamma'] = retrieve_if_not_none(net, 1, 'weight')
    net_params[scope + '/conv1/norm/beta'] = retrieve_if_not_none(net, 1, 'bias')
    net_params[scope + '/conv1/norm/moving_mean'] = retrieve_if_not_none(net, 1, 'running_mean')
    net_params[scope + '/conv1/norm/moving_variance'] = retrieve_if_not_none(net, 1, 'running_var')

    net_params[scope + '/conv2/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 4, 'weight'))
    net_params[scope + '/conv2/biases'] = retrieve_if_not_none(net, 4, 'bias')

    net_params[scope + '/conv2/norm/gamma'] = retrieve_if_not_none(net, 5, 'weight')
    net_params[scope + '/conv2/norm/beta'] = retrieve_if_not_none(net, 5, 'bias')
    net_params[scope + '/conv2/norm/moving_mean'] = retrieve_if_not_none(net, 5, 'running_mean')
    net_params[scope + '/conv2/norm/moving_variance'] = retrieve_if_not_none(net, 5, 'running_var')

    net_params[scope + '/conv3/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 8, 'weight'))
    net_params[scope + '/conv3/biases'] = retrieve_if_not_none(net, 8, 'bias')

    net_params[scope + '/conv3/norm/gamma'] = retrieve_if_not_none(net, 9, 'weight')
    net_params[scope + '/conv3/norm/beta'] = retrieve_if_not_none(net, 9, 'bias')
    net_params[scope + '/conv3/norm/moving_mean'] = retrieve_if_not_none(net, 9, 'running_mean')
    net_params[scope + '/conv3/norm/moving_variance'] = retrieve_if_not_none(net, 9, 'running_var')

    net_params[scope + '/conv4/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 11, 'weight'))
    net_params[scope + '/conv4/biases'] = retrieve_if_not_none(net, 11, 'bias')

    net_params[scope + '/conv4/norm/gamma'] = retrieve_if_not_none(net, 12, 'weight')
    net_params[scope + '/conv4/norm/beta'] = retrieve_if_not_none(net, 12, 'bias')
    net_params[scope + '/conv4/norm/moving_mean'] = retrieve_if_not_none(net, 12, 'running_mean')
    net_params[scope + '/conv4/norm/moving_variance'] = retrieve_if_not_none(net, 12, 'running_var')

    net_params[scope + '/conv5/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 14, 'weight'))
    net_params[scope + '/conv5/biases'] = retrieve_if_not_none(net, 14, 'bias')

    net_params[scope + '/conv5/norm/gamma'] = retrieve_if_not_none(net, 15, 'weight')
    net_params[scope + '/conv5/norm/beta'] = retrieve_if_not_none(net, 15, 'bias')
    net_params[scope + '/conv5/norm/moving_mean'] = retrieve_if_not_none(net, 15, 'running_mean')
    net_params[scope + '/conv5/norm/moving_variance'] = retrieve_if_not_none(net, 15, 'running_var')

    net_params[scope + '/conv6/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 18, 'weight'))
    net_params[scope + '/conv6/biases'] = retrieve_if_not_none(net, 18, 'bias')

    net_params[scope + '/conv6/norm/gamma'] = retrieve_if_not_none(net, 19, 'weight')
    net_params[scope + '/conv6/norm/beta'] = retrieve_if_not_none(net, 19, 'bias')
    net_params[scope + '/conv6/norm/moving_mean'] = retrieve_if_not_none(net, 19, 'running_mean')
    net_params[scope + '/conv6/norm/moving_variance'] = retrieve_if_not_none(net, 19, 'running_var')

    net_params[scope + '/conv7/weights'] = transpose_if_not_none(retrieve_if_not_none(net, 21, 'weight'))
    net_params[scope + '/conv7/biases'] = retrieve_if_not_none(net, 21, 'bias')

    net_params[scope + '/conv7/norm/gamma'] = retrieve_if_not_none(net, 22, 'weight')
    net_params[scope + '/conv7/norm/beta'] = retrieve_if_not_none(net, 22, 'bias')
    net_params[scope + '/conv7/norm/moving_mean'] = retrieve_if_not_none(net, 22, 'running_mean')
    net_params[scope + '/conv7/norm/moving_variance'] = retrieve_if_not_none(net, 22, 'running_var')

    if num_classes is None:
        net_params[scope + '/conv8a/weights'] = transpose_if_not_none(retrieve_if_not_none(sub_net, 0, 'weight'))
        net_params[scope + '/conv8a/biases'] = retrieve_if_not_none(sub_net, 0, 'bias')

        net_params[scope + '/conv8b/weights'] = transpose_if_not_none(retrieve_if_not_none(sub_net, 1, 'weight'))
        net_params[scope + '/conv8b/biases'] = retrieve_if_not_none(sub_net, 1, 'bias')

    return net_params
