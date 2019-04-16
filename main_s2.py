import tensorflow as tf

from datetime import datetime
from codebase.loggers import Logger
from codebase.models.audition import HearModel
from codebase.models.audition import SoundNet5Model
from codebase.models.audition import DualCamHybridModel
from codebase.trainers import TwoStreamsTrainer
from codebase.data import ActionsDataLoader as DataLoader

flags = tf.app.flags
flags.DEFINE_string('mode', None, 'Execution mode, it can be either \'train\' or \'test\'')
flags.DEFINE_string('student_model', None, 'Model type, it can be one of \'SoundNet5\', or \'HearNet\'')
flags.DEFINE_string('train_file', None, 'Path to the plain text file for the training set')
flags.DEFINE_string('valid_file', None, 'Path to the plain text file for the validation set')
flags.DEFINE_string('test_file', None, 'Path to the plain text file for the testing set')
flags.DEFINE_string('exp_name', None, 'Name of the experiment')
flags.DEFINE_string('student_init_checkpoint', None, 'Checkpoint file for student model intialization')
flags.DEFINE_string('teacher_restore_checkpoint', None, 'Checkpoint file for teacher model restoring')
flags.DEFINE_string('restore_checkpoint', None, 'Checkpoint file for session restoring')
flags.DEFINE_integer('batch_size', 64, 'Size of the mini-batch')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('display_freq', 10, 'How often must be shown training results')
flags.DEFINE_integer('num_epochs', 10, 'Number of iterations through dataset')
flags.DEFINE_integer('sample_length', 5, 'Length in seconds of a sequence sample')
flags.DEFINE_integer('number_of_crops', 20, 'Number of crops')
flags.DEFINE_integer('buffer_size', 1, 'Size of pre-fetch buffer')
flags.DEFINE_float('alpha_value', 0.0, 'Importance of hallucination loss')
flags.DEFINE_float('lambda_value', 0.5, 'Importance of distillation loss')
flags.DEFINE_float('temperature_value', 1.0, 'Scaling value for teacher activations (defaults to no scaling)')
flags.DEFINE_string('log_dir', None, 'Directory for storing logs')
flags.DEFINE_string('checkpoint_dir', None, 'Directory for storing models')
flags.DEFINE_bool('normalize', False, 'Normalize spectrogram')
flags.DEFINE_boolean('save_best_only', True, 'Flag to indicate whether to save only the best model')
FLAGS = flags.FLAGS

slim = tf.contrib.slim


def main(_):
    # Instantiate logger
    logger = Logger(FLAGS.log_dir, FLAGS.exp_name)

    # Create training and validation data loaders
    print('{}: {} - Creating data loaders'.format(datetime.now(), FLAGS.exp_name))

    build_spectrogram = FLAGS.student_model == 'HearNet'

    with tf.device('/cpu:0'):

        if FLAGS.train_file is None:
            train_data = None
        else:
            train_data = DataLoader(FLAGS.train_file, 'training', FLAGS.batch_size,
                                    sample_length=FLAGS.sample_length,
                                    number_of_crops=FLAGS.number_of_crops, buffer_size=FLAGS.buffer_size,
                                    normalize=FLAGS.normalize, build_spectrogram=build_spectrogram,
                                    shuffle=True, modalities=[0, 1])

        if FLAGS.valid_file is None:
            valid_data = None
        else:
            valid_data = DataLoader(FLAGS.valid_file, 'inference', FLAGS.batch_size,
                                    sample_length=FLAGS.sample_length,
                                    normalize=FLAGS.normalize, build_spectrogram=build_spectrogram,
                                    buffer_size=FLAGS.buffer_size, shuffle=False, modalities=[0, 1])

        if FLAGS.test_file is None:
            test_data = None
        else:
            test_data = DataLoader(FLAGS.test_file, 'inference', FLAGS.batch_size,
                                   sample_length=FLAGS.sample_length,
                                   normalize=FLAGS.normalize, build_spectrogram=build_spectrogram,
                                   buffer_size=FLAGS.buffer_size, shuffle=False, modalities=[0, 1])

    # Build teacher model
    print('{}: {} - Building teacher model'.format(datetime.now(), FLAGS.exp_name))

    with tf.device('/gpu:0'):
        teacher_model = DualCamHybridModel(input_shape=[36, 48, 12], num_classes=14)

    # Build audition model
    print('{}: {} - Building student model'.format(datetime.now(), FLAGS.exp_name))

    with tf.device('/gpu:0'):

        if FLAGS.student_model == 'SoundNet5':
            student_model = SoundNet5Model(input_shape=[22050 * 5, 1, 1], num_classes=14)
        elif FLAGS.student_model == 'HearNet':
            student_model = HearModel(input_shape=[500, 1, 257], num_classes=14)
        else:
            # Not necessary but set model to None to avoid warning about using unassigned local variable
            student_model = None
            raise ValueError('Unknown model type')

    # Build trainer
    print('{}: {} - Building trainer'.format(datetime.now(), FLAGS.exp_name))
    trainer = TwoStreamsTrainer(teacher_model, student_model, logger, display_freq=FLAGS.display_freq,
                                learning_rate=FLAGS.learning_rate,
                                lambda_value=FLAGS.lambda_value,
                                temperature_value=FLAGS.temperature_value,
                                num_epochs=FLAGS.num_epochs)

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
    flags.mark_flags_as_required(['mode', 'exp_name', 'teacher_restore_checkpoint'])
    tf.app.run()

