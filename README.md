This repository contains code for the paper:

Audio-Visual Model Distillation Using Acoustic Images. [arXiv, 2019](https://arxiv.org/abs/1904.07933).

## Requirements

- Python 2
- TensorFlow 1.4.0 >=
- LibROSA 0.6.1 >=

## Contents

We provide several scripts and a package with all the necessary code for training and testing our model. The code is organized in several folders and a couple of main scripts as follows:

- The `codebase` folder contains several sub-packages with common code used to train our models.

- The `utils` folder contains several utility functions and some scripts to manipulate the data of our audio-visually indicated actions dataset.

- The `main_s1` script is used for training and testing the student and teacher networks from action labels.

- The `main_s2` script is used for training and testing the student networks from the teachers soft predictions and the action labels.

## Preparing the dataset

First we need to download the dataset from the
[project's website](https://pavis.iit.it/datasets/audio-visually-indicated-actions-dataset) following the instructions
described therein. The dataset is delivered as a single compressed zip file.

Once downloaded and decompressed, the data has to be converted into TensorFlow's native
[TFRecord](https://www.tensorflow.org/api_docs/python/python_io#tfrecords-format-details) format. Each TFRecord
will contain a [TF-SequenceExample](https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/core/example/example.proto)
protocol buffer with 1 second of data from the three modalities, video images, raw audio, and acoustic images.
It also contains labels data for location, subject, and action.

Depending on the networks you want to train, you might only need data from some of the modalities. It is possible to do
this with the aid of the `convert_data.py` utility script. To do this provide the location of the
dataset and choose the modalities you want to include, 0: acoustic images, 1: raw audio, and 2: video images. For a full
description of the available options, please run the script with the `--help` option. Below we demonstrate how to
generate TFRecords for the single modalities and for all modalities:

```shell
# location where dataset was decompressed
DATA_DIR='./dataset/'

# generate TFRecords only with acoustic images
python utils/convert_data.py "$DATA_DIR/matfiles/" "$DATA_DIR/sync/" "$DATA_DIR/tfrecords/" --modalities 0

# generate TFRecords only with raw audio
python utils/convert_data.py "$DATA_DIR/matfiles/" "$DATA_DIR/sync/" "$DATA_DIR/tfrecords/" --modalities 1

# generate TFRecords only with video images
python utils/convert_data.py "$DATA_DIR/matfiles/" "$DATA_DIR/sync/" "$DATA_DIR/tfrecords/" --modalities 2

# generate TFRecords for all modalities
python utils/convert_data.py "$DATA_DIR/matfiles/" "$DATA_DIR/sync/" "$DATA_DIR/tfrecords/" --modalities 0 1 2
```

When the script has finished running, you will find several TFRecord files created:

```shell
$ cd "$DATA_DIR/tfrecords/"
$ find ./ -name "*.tfrecord" | sort
./Location_01/Subject_01/Action_001/Data_001.tfrecord
...
...
...
./Location_03/Subject_09/Action_014/Data_030.tfrecord
```

These files represent the full dataset sharded over 30 files per location-subject-action combination. The mapping from
location and action labels to class names is as follows:

```
Location_01: anechoic chamber
Location_02: open space office
Location_03: outdoor terrace

Action_001: clapping
Action_002: snapping fingers
Action_003: speaking
Action_004: whistling
Action_005: playing kendama
Action_006: clicking
Action_007: typing
Action_008: knocking
Action_009: hammering
Action_010: peanut breaking
Action_011: paper ripping
Action_012: plastic crumpling
Action_013: paper shaking
Action_014: stick dropping
```

Finally we need to split the dataset, to do so we use the `generate_lists.py` utility script to generate some plain text
list files with the files for each split. We split files in three modes, 0: training, validation and test. 1: training and
validation, 2: no splitting. It is also possible to choose the location (scenario) to split, 1: anechoic chamber, 2:
open space office, and 3: outdoor terrace, None: all scenarios. For a full reference of the available options, please
run the script with the `--help` option. Below it is shown how to generate different dataset splittings:

```shell
# Split dataset into training/validation/test for 1st location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 0 --location 1

# Split dataset into training/validation/test for 2nd location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 0 --location 2

# Split dataset into training/validation/test for 3rd location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 0 --location 3

# Split full dataset into training/validation/test
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 0

# Use all data from 1st location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 2 --location 1

# Use all data from 2nd location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 2 --location 2

# Use all data from 3rd location
python utils/generate_lists.py "${DATA_DIR}/tfrecords/" --num_samples 10 --split_mode 2 --location 3
```

In all cases we generate 10 seconds long sequences (`--num_samples 10`), i.e. we divide the full 30 seconds long
sequences in three parts of equal length and consider some for training/validation/test correspondingly. The above
commands generate the following folders:

```shell
$ cd "${DATA_DIR}/tfrecords/"
$ find ./ -name "*.txt"
./10_seconds_location_3/lists/validation.txt
./10_seconds_location_3/lists/testing.txt
./10_seconds_location_3/lists/training.txt
./10_seconds_location_3/lists/dataset.txt
./10_seconds_location_2/lists/validation.txt
./10_seconds_location_2/lists/testing.txt
./10_seconds_location_2/lists/training.txt
./10_seconds_location_2/lists/dataset.txt
./10_seconds/lists/validation.txt
./10_seconds/lists/testing.txt
./10_seconds/lists/training.txt
./10_seconds_location_1/lists/validation.txt
./10_seconds_location_1/lists/testing.txt
./10_seconds_location_1/lists/training.txt
./10_seconds_location_1/lists/dataset.txt
```

## Pre-trained Models

Some of our teacher and student models are fine-tuned from pre-trained versions. We thus need to download them and place
them in an appropriate folder as follows:

```shell
# location where checkpoints are stored
CKPTS_DIR='./checkpoints/'

# Download and uncompress SoundNet 5-layers pre-trained model
wget -N http://data.csail.mit.edu/soundnet/soundnet_models_public.zip -P "$CKPTS_DIR"
unzip -jo "$CKPTS_DIR/soundnet_models_public.zip" 'soundnet_models_public/soundnet5_final.t7' -d "$CKPTS_DIR/soundnet/"

# Download and uncompress ResNet-50 pre-trained model
wget -N http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz -P "$CKPTS_DIR"
tar -xvf "$CKPTS_DIR/resnet_v1_50_2016_08_28.tar.gz" -C "$CKPTS_DIR/resnet/"
```
