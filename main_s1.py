import tensorflow as tf

from datetime import datetime
from codebase.loggers import Logger
from codebase.models.vision import ResNet50Model
from codebase.models.vision import ResNet50TemporalModel
from codebase.models.audition import HearModel
from codebase.models.audition import DualCamModel
from codebase.models.audition import SoundNet5Model
from codebase.models.audition import DualCamHybridModel
from codebase.models.multimodal import AVModel
from codebase.trainers import OneStreamTrainer
from codebase.data import ActionsDataLoader as DataLoader

flags = tf.app.flags
flags.DEFINE_string('mode', None, 'Execution mode, it can be either \'train\' or \'test\'')
flags.DEFINE_string('model', None, 'Model type, it can be one of \'SeeNet\', \'ResNet50\', \'TemporalResNet50\', '
                                   '\'DualCamNet\', \'DualCamHybridNet\', \'SoundNet5\', or \'HearNet\'')
flags.DEFINE_string('train_file', None, 'Path to the plain text file for the training set')
flags.DEFINE_string('valid_file', None, 'Path to the plain text file for the validation set')
flags.DEFINE_string('test_file', None, 'Path to the plain text file for the testing set')
flags.DEFINE_string('exp_name', None, 'Name of the experiment')
flags.DEFINE_string('init_checkpoint', None, 'Checkpoint file for model intialization')
flags.DEFINE_string('restore_checkpoint', None, 'Checkpoint file for session restoring')
flags.DEFINE_integer('batch_size', 64, 'Size of the mini-batch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('display_freq', 10, 'How often must be shown training results')
flags.DEFINE_integer('num_epochs', 10, 'Number of iterations through dataset')
flags.DEFINE_integer('sample_length', 5, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('number_of_crops', 20, 'Number of crops')
flags.DEFINE_integer('buffer_size', 1, 'Size of pre-fetch buffer')
flags.DEFINE_string('log_dir', None, 'Directory for storing logs')
flags.DEFINE_string('checkpoint_dir', None, 'Directory for storing models')
flags.DEFINE_boolean('temporal_pooling', False, 'Flag to indicate whether to use average pooling over time')
flags.DEFINE_boolean('normalize', False, 'Flag to indicate whether to use normalization')
flags.DEFINE_boolean('save_best_only', True, 'Flag to indicate whether to save only the best model')
FLAGS = flags.FLAGS


def main(_):
    # Instantiate logger
    logger = Logger('{}/{}'.format(FLAGS.log_dir, FLAGS.exp_name))

    # Create data loaders according to the received program arguments
    print('{}: {} - Creating data loaders'.format(datetime.now(), FLAGS.exp_name))

    random_pick = FLAGS.model == 'TemporalResNet50'
    build_spectrogram = FLAGS.model == 'HearNet'
    modalities = []

    if FLAGS.model == 'DualCamNet' or FLAGS.model == 'DualCamHybridNet':
        modalities.append(0)
    elif FLAGS.model == 'SoundNet5' or FLAGS.model == 'HearNet':
        modalities.append(1)
    elif FLAGS.model == 'SeeNet' or FLAGS.model == 'ResNet50' or FLAGS.model == 'TemporalResNet50':
        modalities.append(2)
    elif FLAGS.model == 'AVModel':
        modalities.append(0)
        modalities.append(2)

    with tf.device('/cpu:0'):

        if FLAGS.train_file is None:
            train_data = None
        else:
            train_data = DataLoader(FLAGS.train_file, 'training', FLAGS.batch_size,
                                    sample_length=FLAGS.sample_length,
                                    number_of_crops=FLAGS.number_of_crops, buffer_size=FLAGS.buffer_size,
                                    shuffle=True, normalize=FLAGS.normalize, random_pick=random_pick,
                                    build_spectrogram=build_spectrogram, modalities=modalities)

        if FLAGS.valid_file is None:
            valid_data = None
        else:
            valid_data = DataLoader(FLAGS.valid_file, 'inference', FLAGS.batch_size,
                                    sample_length=FLAGS.sample_length,
                                    buffer_size=FLAGS.buffer_size, shuffle=False, normalize=FLAGS.normalize,
                                    random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities)

        if FLAGS.test_file is None:
            test_data = None
        else:
            test_data = DataLoader(FLAGS.test_file, 'inference', FLAGS.batch_size,
                                   sample_length=FLAGS.sample_length,
                                   buffer_size=FLAGS.buffer_size, shuffle=False, normalize=FLAGS.normalize,
                                   random_pick=random_pick, build_spectrogram=build_spectrogram, modalities=modalities)

    # Build model
    print('{}: {} - Building model'.format(datetime.now(), FLAGS.exp_name))

    with tf.device('/gpu:0'):

        if FLAGS.model == 'ResNet50':
            model = ResNet50Model(input_shape=[224, 224, 3], num_classes=14)
        elif FLAGS.model == 'TemporalResNet50':
            model = ResNet50TemporalModel(input_shape=[224, 224, 3], num_classes=14, nr_frames=5)
        elif FLAGS.model == 'DualCamNet':
            model = DualCamModel(input_shape=[36, 48, 12], num_classes=14)
        elif FLAGS.model == 'DualCamHybridNet':
            model = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=14)
        elif FLAGS.model == 'SoundNet5':
            model = SoundNet5Model(input_shape=[22050 * 5, 1, 1], num_classes=14)
        elif FLAGS.model == 'HearNet':
            model = HearModel(input_shape=[500, 1, 257], num_classes=14)
        elif FLAGS.model == 'AVModel':
            model = AVModel(num_classes=14)
        else:
            # Not necessary but set model to None to avoid warning about using unassigned local variable
            model = None
            raise ValueError('Unknown model type')

    # Build trainer
    print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))
    trainer = OneStreamTrainer(model, logger, display_freq=FLAGS.display_freq, learning_rate=FLAGS.learning_rate,
                               num_epochs=FLAGS.num_epochs, temporal_pooling=FLAGS.temporal_pooling)

    if FLAGS.mode == 'train':
        # Train model
        print('{}: {} - Training started'.format(datetime.now(), FLAGS.exp_name))
        trainer.train(train_data=train_data, valid_data=valid_data)
    elif FLAGS.mode == 'test':
        # Test model
        print('{}: {} - Testing started'.format(datetime.now(), FLAGS.exp_name))
        trainer.test(test_data=test_data)
    else:
        raise ValueError('Unknown execution mode')


if __name__ == '__main__':
    flags.mark_flags_as_required(['mode', 'model', 'exp_name'])
    tf.app.run()
