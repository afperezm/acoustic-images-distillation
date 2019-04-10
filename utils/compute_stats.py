from __future__ import division
from datetime import datetime

import argparse
import tensorflow as tf
import numpy as np
import os
import sys

from codebase.data import ActionsDataLoader as DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', help='Directory holding generated lists and computed statistics', type=str)
parser.add_argument('--batch_size', help='Size of the mini-batch', type=int, default=64)
parsed_args = parser.parse_args()

root_dir = parsed_args.root_dir
batch_size = parsed_args.batch_size
num_channels = 257

lists_dir = '{}/lists'.format(root_dir)
stats_dir = '{}/stats'.format(root_dir)

build_spectrogram = True
total_length = 30               # full sequence length 30 seconds
sample_length = 5               # sample 5 seconds sequences
modality = 1                    # spectrogram
input_shape = [500, 1, 257]     # spectrogram tensor shape
reduce_axis = (0, 1, 2)         # reduce NHWC

if os.path.exists(stats_dir) and os.listdir(stats_dir):
    print("Statistics already computed!")
    sys.exit(0)

if not os.path.exists(lists_dir):
    os.makedirs(lists_dir)

train_file = '{}/training.txt'.format(lists_dir)

with tf.device('/cpu:0'):
    train_data = DataLoader(train_file, 'inference', batch_size,
                            sample_length=sample_length,
                            build_spectrogram=build_spectrogram,
                            modalities=[modality])
    iterator = train_data.data.make_one_shot_iterator()
    next_batch = iterator.get_next()

data_size = train_data.data_size

global_min_value = np.full(num_channels, np.inf, np.float64)
global_max_value = np.full(num_channels, 0.0, np.float64)
global_sum_value = np.full(num_channels, 0.0, np.float64)
global_sum_squared_value = np.full(num_channels, 0.0, np.float64)

# http://mathcentral.uregina.ca/qq/database/qq.09.02/carlos1.html
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

global_min_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_max_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_sum_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))
global_sum_squared_curr = tf.placeholder(dtype=tf.float64, shape=(num_channels,))

batch_min = tf.cast(tf.reduce_min(next_batch[modality], axis=reduce_axis), dtype=tf.float64)
global_min = tf.reduce_min(tf.stack([global_min_curr, batch_min]), axis=0)

batch_max = tf.cast(tf.reduce_max(next_batch[modality], axis=reduce_axis), dtype=tf.float64)
global_max = tf.reduce_max(tf.stack([global_max_curr, batch_max]), axis=0)

batch_sum = tf.cast(tf.reduce_sum(next_batch[modality], axis=reduce_axis), dtype=tf.float64)
global_sum = tf.add(global_sum_curr, batch_sum)

batch_sum_squared = tf.cast(tf.reduce_sum(tf.square(next_batch[modality]), axis=reduce_axis), dtype=tf.float64)
global_sum_squared = tf.add(global_sum_squared_curr, batch_sum_squared)

batch_count = 0

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:

    print('{} Starting'.format(datetime.now()))

    while True:
        try:
            start_time = datetime.now()
            print('{} Processing batch {}'.format(start_time, batch_count + 1))
            global_min_value, global_max_value, global_sum_value, global_sum_squared_value = session.run(
                [global_min, global_max, global_sum, global_sum_squared],
                feed_dict={global_min_curr: global_min_value,
                           global_max_curr: global_max_value,
                           global_sum_curr: global_sum_value,
                           global_sum_squared_curr: global_sum_squared_value})
            end_time = datetime.now()
            print('{} Completed in {} seconds'.format(end_time, (end_time - start_time).total_seconds()))
        except tf.errors.OutOfRangeError:
            print('{} Cancelled'.format(datetime.now()))
            break
        batch_count += 1

    print('{} Completed'.format(datetime.now()))

if not os.path.exists(stats_dir):
    os.makedirs(stats_dir)

np.save('{}/global_min.npy'.format(stats_dir), global_min_value.astype(np.float32))
np.save('{}/global_max.npy'.format(stats_dir), global_max_value.astype(np.float32))
np.save('{}/global_sum.npy'.format(stats_dir), global_sum_value.astype(np.float32))
np.save('{}/global_sum_squared.npy'.format(stats_dir), global_sum_squared_value.astype(np.float32))

n = data_size * reduce(lambda x, y: x * y, input_shape)

global_mean = global_sum_value / n
global_var = (global_sum_squared_value - (global_sum_value ** 2) / n) / n
global_std_dev = np.sqrt(global_var)

np.save('{}/global_mean.npy'.format(stats_dir), global_mean.astype(np.float32))
np.save('{}/global_std_dev.npy'.format(stats_dir), global_std_dev.astype(np.float32))
