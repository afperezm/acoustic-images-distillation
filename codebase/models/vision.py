from tensorflow.contrib.slim.nets import resnet_v1

import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_temporal as resnet_v2


class ResNet50Model(object):

    def __init__(self, input_shape=None, num_classes=14):

        self.scope = 'resnet_v1_50'

        self.num_classes = num_classes

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope + '/logits')

    def init_model(self, session, checkpoint_file):
        """
        Initializes ResNet-50 network parameters using slim.
        """

        # Restore all model variables up to logits layer (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['logits'])
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

        # Initialize logits layer variables
        logits_variables = slim.get_model_variables(self.scope + '/logits')
        logits_init_op = tf.variables_initializer(logits_variables)
        session.run(logits_init_op)

    def _build_model(self):
        """
        Builds a ResNet-50 network using slim.
        """

        visual_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='visual_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=5e-4)):
            output, network = resnet_v1.resnet_v1_50(visual_images, num_classes=self.num_classes,
                                                     is_training=is_training)

        output = tf.squeeze(output, [1, 2])

        network.update({
            'input': visual_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        return output, network


class ResNet50TemporalModel(object):
    def __init__(self, input_shape=None, num_classes=14, nr_frames=5):
        self.scope = 'resnet_v1_50'

        self.num_classes = num_classes

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]
        self.nr_frames = nr_frames
        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope + '/logits')

    def init_model(self, session, checkpoint_file):
        """
        Initializes ResNet-50 network parameters using slim.
        """

        model_variables = slim.get_model_variables(self.scope)

        # Initialize model variables excluding temporal and logits layers
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['conv_temp', 'logits'])
        model_init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        model_init_fn(session)

        # Initialize temporal layer variables
        temporal_variables = slim.filter_variables(model_variables, include_patterns=['conv_temp'])
        temporal_init_op = tf.variables_initializer(temporal_variables)
        session.run(temporal_init_op)

        # Initialize logits layer variables
        logits_variables = slim.filter_variables(model_variables, include_patterns=['logits'])
        logits_init_op = tf.variables_initializer(logits_variables)
        session.run(logits_init_op)

    def _build_model(self):
        """
        Builds a ResNet-50 network using slim.
        """
        visual_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels],
                                       name='visual_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=0.997)):
            output, network = resnet_v2.resnet_one_stream_main(visual_images,
                                                               nr_frames=self.nr_frames,
                                                               num_classes=self.num_classes,
                                                               is_training=is_training,
                                                               scope=self.scope)

            # predictions for each video are the avg of frames' predictions
            # TRAIN ###############################
            output = tf.reshape(output, [-1, self.nr_frames, self.num_classes])
            output = tf.reduce_mean(output, axis=1)

        network.update({
            'input': visual_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        return output, network
