import dualcamnet
import hearnet
import shared
import soundnet
import tensorflow as tf
import tensorflow.contrib.slim as slim

from collections import OrderedDict

flags = tf.app.flags
FLAGS = flags.FLAGS


class DualCamModel(object):

    def __init__(self, mode='train', input_shape=None, num_classes=14, num_frames=12, sample_length=1):

        self.scope = 'DualCamNet'
        self.mode = mode
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.sample_length = sample_length

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope)

    def init_model(self, session, checkpoint_file):
        """
        Initializes network parameters from ckpt.
        """

        # Restore all model variables up to fc2 (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['fc2'])
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

        # Initialize fc2 variables
        fc2_variables = slim.get_model_variables(self.scope + '/fc2')
        fc2_init_op = tf.variables_initializer(fc2_variables)
        session.run(fc2_init_op)

    def _build_model(self):
        """
        Builds the model.
        """

        acoustic_images = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels],
                                         name='acoustic_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        output, network = dualcamnet.dualcamnet_v2(acoustic_images,
                                                   keep_prob=keep_prob,
                                                   is_training=is_training,
                                                   num_classes=self.num_classes,
                                                   num_frames=self.num_frames,
                                                   num_channels=self.channels,
                                                   scope=self.scope)

        network.update({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        return output, network


class DualCamHybridModel(object):

    def __init__(self, input_shape=None, num_classes=14, num_frames=12):

        self.scope = 'DualCamNet'

        self.num_classes = num_classes
        self.num_frames = num_frames

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope)

    def init_model(self, session, checkpoint_file):
        """
        Initializes network parameters from ckpt.
        """

        # Restore all model variables up to fc2 (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['fc2'])
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

        # Initialize fc2 variables
        fc2_variables = slim.get_model_variables(self.scope + '/fc2')
        fc2_init_op = tf.variables_initializer(fc2_variables)
        session.run(fc2_init_op)

    def _build_model(self):
        """
        Builds the model.
        """

        acoustic_images = tf.placeholder(tf.float32,
                                         [None, self.height, self.width, self.channels],
                                         name='acoustic_images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': acoustic_images,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        dualcamnet_output, dualcamnet_end_points = dualcamnet.dualcamnet_v2(acoustic_images,
                                                                            keep_prob=keep_prob,
                                                                            is_training=is_training,
                                                                            num_classes=None,
                                                                            num_frames=self.num_frames,
                                                                            num_channels=self.channels,
                                                                            scope=self.scope)

        end_points.update(dualcamnet_end_points)

        sharednet_output, sharednet_end_points = shared.shared_net(dualcamnet_output,
                                                                   num_classes=self.num_classes,
                                                                   is_training=is_training,
                                                                   keep_prob=keep_prob,
                                                                   spatial_squeeze=True,
                                                                   scope=self.scope)

        end_points.update(sharednet_end_points)

        return sharednet_output, end_points


class HearModel(object):

    def __init__(self, input_shape=None, num_classes=14):

        self.scope = 'hear_net'

        self.num_classes = num_classes

        self.height = input_shape[0]
        self.width = input_shape[1]
        self.channels = input_shape[2]

        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope)

    def init_model(self, session, checkpoint_file):
        """
        Initializes network parameters from ckpt.
        """

        # Restore all model variables up to fc2 (excluded)
        model_variables = slim.get_model_variables(self.scope)
        variables_to_restore = slim.filter_variables(model_variables, exclude_patterns=['fc2'])
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
        init_fn(session)

        # Initialize fc2 variables
        fc2_variables = slim.get_model_variables(self.scope + '/fc2')
        fc2_init_op = tf.variables_initializer(fc2_variables)
        session.run(fc2_init_op)

    def _build_model(self):
        """
        Builds the model.
        """

        spectrogram = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='spectrogram')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': spectrogram,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        with slim.arg_scope(hearnet.build_arg_scope(weight_decay=5e-4)):
            hearnet_output, hearnet_end_points = hearnet.build_network(spectrogram,
                                                                       num_classes=self.num_classes,
                                                                       is_training=is_training,
                                                                       keep_prob=keep_prob,
                                                                       scope=self.scope)

        end_points.update(hearnet_end_points)

        shared_net_output, shared_net_end_points = shared.shared_net(hearnet_output,
                                                                     num_classes=self.num_classes,
                                                                     is_training=is_training,
                                                                     keep_prob=keep_prob,
                                                                     spatial_squeeze=True,
                                                                     scope=self.scope)

        end_points.update(shared_net_end_points)

        return shared_net_output, end_points


class SoundNet5Model(object):

    def __init__(self, input_shape=None, num_classes=14):

        self.scope = 'SoundNet'

        self.num_classes = num_classes

        self.height = input_shape[0]    # 22050 * 5
        self.width = input_shape[1]     # 1
        self.channels = input_shape[2]  # 1

        self.output, self.network = self._build_model()

        self.train_vars = slim.get_trainable_variables(self.scope)

    def init_model(self, session, checkpoint_file):
        """
        Initializes network parameters either from ckpt or t7.
        """

        if checkpoint_file.lower().endswith('.t7'):
            # Load network parameters from t7 into a dict
            net_params = soundnet.soundnet5_model_params(model_filename=checkpoint_file, num_classes=self.num_classes,
                                                         scope=self.scope)
            # Restore all model variables up to conv5 (excluded)
            init_fn = slim.assign_from_values_fn(net_params)
            init_fn(session)
        else:
            # Restore all model variables up to conv5 (excluded)
            model_variables = slim.get_variables(self.scope)
            variables_to_restore = slim.filter_variables(model_variables,
                                                         exclude_patterns=['conv5', 'conv6', 'fc1', 'fc2'])
            init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)
            init_fn(session)

        # Initialize conv5, conv6, fc1 and fc2 variables
        init_op = tf.variables_initializer(slim.get_variables(self.scope + '/conv5') +
                                           slim.get_variables(self.scope + '/conv6') +
                                           slim.get_model_variables(self.scope + '/fc1') +
                                           slim.get_model_variables(self.scope + '/fc2'))
        session.run(init_op)

    def _build_model(self):
        """
        Builds the model.
        """

        sound_input = tf.placeholder(tf.float32, [None, self.height, self.width, self.channels], name='sound_input')
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        end_points = OrderedDict({
            'input': sound_input,
            'is_training': is_training,
            'keep_prob': keep_prob
        })

        with slim.arg_scope(soundnet.soundnet_arg_scope(is_training=is_training, weight_decay=5e-4)):
            soundnet_output, soundnet_end_points = soundnet.soundnet5(sound_input,
                                                                      num_classes=self.num_classes,
                                                                      scope=self.scope)

        # soundnet_output, soundnet_end_points = soundnet.soundnet5(sound_input,
        #                                                           num_classes=self.num_classes,
        #                                                           is_training=is_training,
        #                                                           keep_prob=keep_prob,
        #                                                           scope=self.scope)

        end_points.update(soundnet_end_points)

        shared_net_output, shared_net_end_points = shared.shared_net(soundnet_output,
                                                                     num_classes=self.num_classes,
                                                                     is_training=is_training,
                                                                     keep_prob=keep_prob,
                                                                     spatial_squeeze=True,
                                                                     scope=self.scope)

        end_points.update(shared_net_end_points)

        return shared_net_output, end_points
