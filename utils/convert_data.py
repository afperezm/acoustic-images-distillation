import argparse
import cv2
import glob
import numpy as np
import os
import re
import tensorflow as tf

from collections import namedtuple
from datetime import datetime
from scipy import io as spio
from utils import str2dir

Image = namedtuple('Image', 'rows cols depth data')
Audio = namedtuple('Audio', 'mics samples data')

_BROKEN_MICS_IDX = [79, 105]
_NUMBER_OF_MICS = 128
_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12
_MIN_LENGTH = 30


def _read_acoustic_image(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))

    img_raw = spio.loadmat(filename)['MFCC'].astype('<f4')
    pad_width = [(before, 0) for before in np.array([36, 48, 12]) - np.array(img_raw.shape)]
    img_padded = np.pad(img_raw, pad_width=pad_width, mode='constant', constant_values=0)

    rows = img_padded.shape[0]
    cols = img_padded.shape[1]
    depth = img_padded.shape[2]
    image_serialized = img_padded.tostring()

    return Image(rows=rows, cols=cols, depth=depth, data=image_serialized)


def _read_raw_audio_data(audio_sample_file):
    print('{} - Reading {}'.format(datetime.now(), audio_sample_file))

    audio_data_sample = np.load(audio_sample_file)
    audio_serialized = audio_data_sample.tostring()

    return Audio(mics=audio_data_sample.shape[0], samples=audio_data_sample.shape[1], data=audio_serialized)


def _read_video_frame(filename):
    print('{} - Reading {}'.format(datetime.now(), filename))

    image_raw = cv2.imread(filename)

    rows = image_raw.shape[0]
    cols = image_raw.shape[1]
    depth = image_raw.shape[2]
    image_serialized = image_raw.tostring()

    return Image(rows=rows, cols=cols, depth=depth, data=image_serialized)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Matlab files data set root directory', type=str2dir)
    parser.add_argument('root_raw_dir', help='Synchronized raw files data set root directory', type=str2dir)
    parser.add_argument('out_dir', help='Directory where to store the converted data', type=str2dir)
    parser.add_argument('--modalities', help='Modalities to consider. 0: Audio images. 1: Audio data. 2: Video data.',
                        nargs='*', type=int)
    parsed_args = parser.parse_args()

    root_dir = parsed_args.root_dir
    root_raw_dir = parsed_args.root_raw_dir
    out_dir = parsed_args.out_dir
    modalities = parsed_args.modalities
    include_audio_images = modalities is None or 0 in modalities
    include_audio_data = modalities is None or 1 in modalities
    include_video_data = modalities is None or 2 in modalities
    num_frames = _FRAMES_PER_SECOND * _MIN_LENGTH
    num_samples = _FRAMES_PER_SECOND

    data_dirs = sorted(glob.glob('{}/*/*/*/MFCC_Image/'.format(root_dir)))

    for data_mat_dir in data_dirs:

        splitted_data_dir = data_mat_dir.split('/')
        location = int(filter(re.compile('Location_.*').match, splitted_data_dir)[0].split('_')[1])
        subject = int(filter(re.compile('Subject_.*').match, splitted_data_dir)[0].split('_')[1])
        action = int(filter(re.compile('Action_.*').match, splitted_data_dir)[0].split('_')[1])

        data_raw_audio_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('MFCC_Image', 'audio')
        data_raw_video_dir = data_mat_dir.replace(root_dir, root_raw_dir).replace('MFCC_Image', 'video')

        num_mat_files = len([name for name in os.listdir(data_mat_dir) if name.endswith('.mat')])
        num_raw_audio_files = len([name for name in os.listdir(data_raw_audio_dir) if name.endswith('.npy')])
        num_raw_video_files = len([name for name in os.listdir(data_raw_video_dir) if name.endswith('.jpeg')])

        # Ensure there are the same number of acoustic images and raw audio and video files
        assert num_mat_files == num_raw_audio_files
        assert num_mat_files == num_raw_video_files

        for idx in range(int(num_frames / num_samples)):

            start_index = idx * num_samples

            # Repeat until meeting minimum length of 30 seconds
            if include_audio_images:
                mat_files = ['{}/Data_{}.mat'.format(data_mat_dir, index % num_mat_files + 1) for index in range(start_index, start_index + num_samples)]
                audio_images = [_read_acoustic_image(filename) for filename in mat_files]
            else:
                audio_images = None

            # Repeat until meeting minimum length of 30 seconds
            if include_audio_data:
                raw_audio_files = ['{}/A_{:06d}.npy'.format(data_raw_audio_dir, index % num_raw_audio_files + 1) for index in range(start_index, start_index + num_samples)]
                audio_data = [_read_raw_audio_data(filename) for filename in raw_audio_files]
            else:
                audio_data = None

            # Repeat until meeting minimum length of 30 seconds
            if include_video_data:
                raw_video_files = ['{}/I_{:06d}.jpeg'.format(data_raw_video_dir, index % num_raw_video_files + 1) for index in range(start_index, start_index + num_samples)]
                video_images = [_read_video_frame(filename) for filename in raw_video_files]
            else:
                video_images = None

            out_data_dir = '{}/Location_{:0>2d}/Subject_{:0>2d}/Action_{:0>3d}/'.format(out_dir, location, subject,
                                                                                        action + 1)
            out_filename = '{}/Data_{:0>3d}.tfrecord'.format(out_data_dir, idx + 1)

            if not os.path.exists(out_data_dir):
                os.makedirs(out_data_dir)

            print('{} - Writing {}'.format(datetime.now(), out_filename))

            with tf.python_io.TFRecordWriter(out_filename, options=tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)) as writer:
                # Store audio and video data properties as context features, assuming all sequences are the same size
                feature = {
                    'action': _int64_feature(action),
                    'location': _int64_feature(location - 1),
                    'subject': _int64_feature(subject - 1)
                }
                if include_audio_images:
                    feature.update({
                        'audio_image/height': _int64_feature(audio_images[0].rows),
                        'audio_image/width': _int64_feature(audio_images[0].cols),
                        'audio_image/depth': _int64_feature(audio_images[0].depth)
                    })
                if include_audio_data:
                    feature.update({
                        'audio_data/mics': _int64_feature(audio_data[0].mics),
                        'audio_data/samples': _int64_feature(audio_data[0].samples)
                    })
                if include_video_data:
                    feature.update({
                        'video/height': _int64_feature(video_images[0].rows),
                        'video/width': _int64_feature(video_images[0].cols),
                        'video/depth': _int64_feature(video_images[0].depth),
                    })
                feature_list = {}
                if include_audio_images:
                    feature_list.update({
                        'audio/image': tf.train.FeatureList(feature=[_bytes_feature(audio_image.data) for audio_image in audio_images])
                    })
                if include_audio_data:
                    feature_list.update({
                        'audio/data': tf.train.FeatureList(feature=[_bytes_feature(audio_sample.data) for audio_sample in audio_data])
                    })
                if include_video_data:
                    feature_list.update({
                        'video/image': tf.train.FeatureList(feature=[_bytes_feature(video_image.data) for video_image in video_images])
                    })
                context = tf.train.Features(feature=feature)
                feature_lists = tf.train.FeatureLists(feature_list=feature_list)
                sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                writer.write(sequence_example.SerializeToString())
