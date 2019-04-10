import argparse
import glob
import numpy as np
import os
import re

from utils import str2dir

TRAIN_SET_SIZE = 0.8
VALID_SET_SIZE = 0.1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', help='Dataset root directory', type=str2dir)
    parser.add_argument('--location', nargs='?', type=int, choices=range(1, 4))
    parser.add_argument('--num_samples', help='Number of files to sample continuously.', type=int, default=1)
    parser.add_argument('--split_mode', help='Splitting mode. 0: train/val/test. 1: train/val, 2: all.', type=int, choices=range(3), default=0)
    parsed_args = parser.parse_args()

    files = []

    root_dir = parsed_args.root_dir
    selected_location = parsed_args.location
    num_samples = parsed_args.num_samples
    split_mode = parsed_args.split_mode

    locations = None if selected_location is None else [selected_location]
    data_dirs = sorted(glob.glob('{}/*/*/*/'.format(root_dir)))

    for data_dir in data_dirs:
        splitted_data_dir = data_dir.split('/')
        location = int(filter(re.compile('Location_.*').match, splitted_data_dir)[0].split('_')[1])
        subject = int(filter(re.compile('Subject_.*').match, splitted_data_dir)[0].split('_')[1])
        action = int(filter(re.compile('Action_.*').match, splitted_data_dir)[0].split('_')[1])

        num_files = len([name for name in os.listdir(data_dir) if name.endswith('.tfrecord')])

        if locations is not None and location not in locations:
            continue

        for index in range(num_samples * int(num_files / num_samples)):
            files.append('{}/Data_{:0>3d}.tfrecord'.format(data_dir, index + 1))

    data_set_size = len(files)

    train_set_size = int(data_set_size * TRAIN_SET_SIZE / num_samples)
    validation_set_size = int(data_set_size * VALID_SET_SIZE / num_samples)
    test_set_size = data_set_size - train_set_size - validation_set_size

    assert data_set_size == train_set_size + validation_set_size + test_set_size

    if split_mode == 1:
        train_set_size = train_set_size + validation_set_size
        validation_set_size = test_set_size
        test_set_size = 0
    elif split_mode == 2:
        train_set_size = train_set_size + validation_set_size + test_set_size
        validation_set_size = 0
        test_set_size = 0

    indices = range(0, data_set_size, num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices = indices[:train_set_size]
    validation_indices = indices[train_set_size:train_set_size + validation_set_size]
    test_indices = indices[train_set_size + validation_set_size:]

    train_files = [files[j] for i in train_indices for j in range(i, i + num_samples)]
    validation_files = [files[j] for i in validation_indices for j in range(i, i + num_samples)]
    test_files = [files[j] for i in test_indices for j in range(i, i + num_samples)]

    # Create directory where to store the generated lists
    out_dir = '{}/{}{}/lists'.format(root_dir,
                                       '{}_seconds'.format(num_samples) if num_samples > 1 else '{}_second'.format(
                                           num_samples),
                                       '_location_{}'.format(locations[0]) if locations is not None else '')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if len(train_files) > 0:
        with open('{}/{}.txt'.format(out_dir, 'dataset' if split_mode == 2 else 'training'), 'w') as f:
            for item in train_files:
                f.write("%s\n" % item)

    if len(validation_files) > 0:
        with open('{}/validation.txt'.format(out_dir), 'w') as f:
            for item in validation_files:
                f.write("%s\n" % item)

    if len(test_files) > 0:
        with open('{}/testing.txt'.format(out_dir), 'w') as f:
            for item in test_files:
                f.write("%s\n" % item)
