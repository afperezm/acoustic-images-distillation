from collections import OrderedDict
from tensorflow.contrib.slim.nets import resnet_v1

import dualcamnet
import shared
import tensorflow as tf
import tensorflow.contrib.slim as slim


class AVModel(object):

    def __init__(self, num_classes=None):

        self.scope = 'AVNet'

        self.num_classes = num_classes

        self.video_height = 224
        self.video_width = 224
        self.video_channels = 3

        self.audio_height = 36
        self.audio_width = 48
        self.audio_channels = 12

        self.num_frames = 12

        self.output, self.network = self._build_model()

        self.audio_train_vars = slim.get_trainable_variables('DualCamNet')
        self.video_train_vars = slim.get_trainable_variables('resnet_v1_50/logits')
        self.audio_video_train_vars = slim.get_trainable_variables(self.scope)

        self.train_vars = self.audio_train_vars + self.video_train_vars + self.audio_video_train_vars

    def init_model(self, session, checkpoint_file):
        """
        Initializes DualCamNet and ResNet-50 networks parameters using slim.
        """

        # Initialize audio model variables
        model_variables = slim.get_variables('DualCamNet')
        init_op = tf.variables_initializer(model_variables)
        session.run(init_op)

        # Restore video model layers up to logits (excluded)
        model_variables = slim.get_model_variables('resnet_v1_50')
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['logits'])
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

        # Initialize video model logits variables
        logits_variables = slim.get_model_variables('resnet_v1_50/logits')
        logits_init_op = tf.variables_initializer(logits_variables)
        session.run(logits_init_op)

    def _build_video_network(self, visual_images, is_training):
        """
        Builds a ResNet-50 network using slim.
        """

        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=5e-4)):
            output, network = resnet_v1.resnet_v1_50(visual_images,
                                                     num_classes=1024,
                                                     is_training=is_training)

        return output, network

    def _build_audio_network(self, inputs, keep_prob, is_training, scope='DualCamNet'):
        """
        Builds a DualCamNet network for classification using a 3D temporal convolutional layer with 7x1x1 filters.
        """

        output, network = dualcamnet.dualcamnet_v2(inputs,
                                                   keep_prob=keep_prob,
                                                   is_training=is_training,
                                                   num_classes=None,
                                                   num_frames=self.num_frames,
                                                   num_channels=self.audio_channels,
                                                   scope=scope)

        return output, network

    def _build_model(self):

        video_input = tf.placeholder(tf.float32,
                                     shape=[None, self.video_height, self.video_width, self.video_channels],
                                     name='visual_images')

        audio_input = tf.placeholder(tf.float32,
                                     shape=[None, self.audio_height, self.audio_width, self.audio_channels],
                                     name='sound_input')

        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        video_output, video_network = self._build_video_network(video_input,
                                                                is_training=is_training)
        audio_output, audio_network = self._build_audio_network(audio_input,
                                                                keep_prob=keep_prob,
                                                                is_training=is_training)

        output = tf.concat([video_output, audio_output], 3, name='av_logits')

        shared_net_output, shared_net_end_points = shared.shared_net(output,
                                                                     num_classes=self.num_classes,
                                                                     is_training=is_training,
                                                                     keep_prob=keep_prob,
                                                                     spatial_squeeze=True,
                                                                     scope=self.scope)

        network = OrderedDict(video_network.items() + audio_network.items() + shared_net_end_points.items())

        network.update({
            'video_input': video_input,
            'audio_input': audio_input,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        return shared_net_output, network
