from __future__ import division
from tensorflow.contrib.slim.nets import vgg
from preprocessing import vgg_preprocessing

import librosa
import tensorflow as tf
import numpy as np
import math

flags = tf.app.flags
FLAGS = flags.FLAGS

# The real number of tracks is 128 corresponding to the number of microphones but every TFRecord has only 1 audio track
_NUMBER_OF_TRACKS = 1
_NUMBER_OF_SAMPLES = 1024
_FRAMES_PER_SECOND = 12
_NUMBER_OF_FRAMES = 5
_NUMBER_OF_CHANNELS = 12
_NUM_ACTIONS = 14
_NUM_LOCATIONS = 3
_NUM_SUBJECTS = 9
_IMAGE_SIZE = vgg.vgg_16.default_image_size


class ActionsDataLoader(object):

    def __init__(self, txt_file, mode, batch_size, sample_rate=22050, total_length=10, sample_length=5,
                 number_of_crops=6, buffer_size=1, num_epochs=1, shuffle=False, normalize=False,
                 random_pick=False, build_spectrogram=False, modalities=None):

        self.seed = tf.placeholder(tf.int64, shape=(), name='data_seed')    # epoch number

        self.txt_file = txt_file
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.total_length = total_length
        self.sample_length = sample_length
        self.number_of_crops = number_of_crops

        self.frame_length = 440
        self.frame_step = 220
        self.fft_length = 512

        self.include_audio_images = modalities is None or 0 in modalities
        self.include_audio_data = modalities is None or 1 in modalities
        self.include_video_data = modalities is None or 2 in modalities

        assert txt_file is not None
        assert total_length % sample_length == 0
        assert (self.include_audio_images or self.include_audio_data or self.include_video_data) is True
        # TODO Fix this assertion to check that there are enough samples to provide the required number of crops
        # assert number_of_crops <= total_length - sample_length

        # Load statistics
        if normalize and self.include_audio_data:
            self.global_min, self.global_max = self._load_stats()

        # Retrieve data from the text file (data_size is the number of files inside)
        self.data_size = self._read_txt_file()

        # Convert img_paths into a tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.files = tf.data.Dataset.from_tensor_slices(self.img_paths)

        # Assert the number of available files is divisible by the given full video sequence length
        assert self.data_size % self.total_length == 0

        # Apply data augmentation if required
        if mode == 'training':
            self.num_samples = int(self.data_size / self.total_length) * self.number_of_crops
            self.files = self.files.batch(self.total_length)
            self.files = self.files.flat_map(self._map_func_sample_sequences)
            self.files = self.files.apply(tf.contrib.data.unbatch())
        elif mode == 'inference':
            self.num_samples = int(self.data_size / self.total_length) * int(self.total_length / self.sample_length)
        else:
            raise ValueError('Unknown mode')

        # Shuffle num_samples blocks of files and repeat them num_epochs
        if shuffle:
            self._shuffle_and_repeat_lists(num_epochs, self.num_samples)

        # Create dataset
        data = self.files.flat_map(lambda ds: tf.data.TFRecordDataset(ds, compression_type='GZIP'))

        # Parse dataset
        data = data.map(self._map_func_parse_sequences, num_parallel_calls=4)

        # Prefetch buffer_size batches of elements of the dataset
        data = data.prefetch(buffer_size=buffer_size * batch_size * sample_length)

        # Batch elements in groups of sample_length seconds
        data = data.batch(self.sample_length)
        data = data.map(self._map_func_prepare_sequences)

        # Build waveform for each sampled sequence
        if self.include_audio_data:
            data = data.map(self._map_func_audio_samples_build_wav, num_parallel_calls=4)

        # Build spectrogram for each waveform
        if self.include_audio_data and build_spectrogram:
            data = data.map(self._map_func_audio_samples_build_spectrogram, num_parallel_calls=4)

        # Apply min-max normalization to each spectrogram
        if self.include_audio_data and normalize:
            data = data.map(self._map_func_audio_samples_normalize_spectrograms, num_parallel_calls=4)

        # Pick random frames from video sequence
        if self.include_video_data and random_pick:
            data = data.map(self._map_func_video_images_pick_frames, num_parallel_calls=4)

        # Pre-process video images (resize, crop, subtract mean)
        if self.include_video_data:
            data = data.map(self._map_func_video_images, num_parallel_calls=4)

        # Create batched dataset that repeats only once (dataset iterator should be re-initialized at each epoch)
        data = data.batch(batch_size)

        self.data = data

    def _load_stats(self):
        """Load spectrogram statistics and convert them into tensors."""

        stats_dir = str.join('/', self.txt_file.replace('//', '/').split('/')[:-2] + ['stats'])
        min_value = np.load('{}/global_min.npy'.format(stats_dir))
        max_value = np.load('{}/global_max.npy'.format(stats_dir))

        global_min = tf.tile(
            tf.expand_dims(input=tf.expand_dims(input=tf.convert_to_tensor(min_value), axis=0), axis=0),
            [500, 1, 1]
        )

        global_max = tf.tile(
            tf.expand_dims(input=tf.expand_dims(input=tf.convert_to_tensor(max_value), axis=0), axis=0),
            [500, 1, 1]
        )

        return global_min, global_max

    def _read_txt_file(self):
        """Read the content of the text file and store it into a list."""
        self.img_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_path = line.rstrip('\n')
                self.img_paths.append(img_path)
        return len(self.img_paths)

    def _shuffle_and_repeat_lists(self, num_epochs, num_samples):
        """Shuffle and repeat the list of paths."""
        self.files = self.files.batch(self.sample_length)
        self.files = self.files.shuffle(buffer_size=num_samples, seed=self.seed, reshuffle_each_iteration=True)
        self.files = self.files.repeat(num_epochs)
        self.files = self.files.apply(tf.contrib.data.unbatch())

    def _map_func_parse_sequences(self, sequence_example_proto):
        """Input parser for samples of the training set."""

        context_features = {'action': tf.FixedLenFeature([], tf.int64),
                            'location': tf.FixedLenFeature([], tf.int64),
                            'subject': tf.FixedLenFeature([], tf.int64)}
        sequence_features = {}

        if self.include_audio_images:
            context_features.update({
                'audio_image/height': tf.FixedLenFeature([], tf.int64),
                'audio_image/width': tf.FixedLenFeature([], tf.int64),
                'audio_image/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        if self.include_audio_data:
            context_features.update({
                'audio_data/mics': tf.FixedLenFeature([], tf.int64),
                'audio_data/samples': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'audio/data': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        if self.include_video_data:
            context_features.update({
                'video/height': tf.FixedLenFeature([], tf.int64),
                'video/width': tf.FixedLenFeature([], tf.int64),
                'video/depth': tf.FixedLenFeature([], tf.int64)
            })
            sequence_features.update({
                'video/image': tf.FixedLenSequenceFeature([], dtype=tf.string)
            })

        # Parse single example
        parsed_context_features, parsed_sequence_features = tf.parse_single_sequence_example(sequence_example_proto,
                                                                                             context_features=context_features,
                                                                                             sequence_features=sequence_features)

        action = tf.cast(parsed_context_features['action'], tf.int32)
        location = tf.cast(parsed_context_features['location'], tf.int32)
        subject = tf.cast(parsed_context_features['subject'], tf.int32)

        if self.include_audio_images:
            # Retrieve parsed context features
            audio_height = tf.cast(parsed_context_features['audio_image/height'], tf.int32)
            audio_width = tf.cast(parsed_context_features['audio_image/width'], tf.int32)
            audio_depth = tf.cast(parsed_context_features['audio_image/depth'], tf.int32)
            # Retrieve parsed audio image features
            audio_image_decoded = tf.decode_raw(parsed_sequence_features['audio/image'], tf.float32)
            # Reshape decoded audio image
            audio_image_shape = tf.stack([-1, audio_height, audio_width, audio_depth])
            audio_images = tf.reshape(audio_image_decoded, audio_image_shape)
        else:
            audio_images = tf.zeros([], tf.int32)

        if self.include_audio_data:
            # Retrieve parsed context features
            num_mics = tf.cast(parsed_context_features['audio_data/mics'], tf.int32)
            num_samples = tf.cast(parsed_context_features['audio_data/samples'], tf.int32)
            # Retrieve parsed audio data features
            audio_sample_decoded = tf.decode_raw(parsed_sequence_features['audio/data'], tf.int32)
            # Reshape decoded audio data
            audio_sample_shape = tf.stack([-1, num_mics, num_samples])
            audio_samples = tf.reshape(audio_sample_decoded, audio_sample_shape)
        else:
            audio_samples = tf.zeros([], tf.int32)

        if self.include_video_data:
            # Retrieve parsed video image features
            video_image_decoded = tf.decode_raw(parsed_sequence_features['video/image'], tf.uint8)
            # Retrieve parsed context features
            video_height = tf.cast(parsed_context_features['video/height'], tf.int32)
            video_width = tf.cast(parsed_context_features['video/width'], tf.int32)
            video_depth = tf.cast(parsed_context_features['video/depth'], tf.int32)
            # Reshape decoded video image
            video_image_shape = tf.stack([-1, video_height, video_width, video_depth])
            video_images = tf.reshape(video_image_decoded, video_image_shape)
        else:
            video_images = tf.zeros([], tf.int32)

        return audio_images, audio_samples, video_images, action, location, subject

    def _map_func_sample_sequences(self, files):
        """Input mapper implementing data augmentation (sequences sampling)."""

        # Compute indices of random crops
        shapes = tf.constant(self.total_length, dtype=tf.int32, shape=[self.number_of_crops])
        sizes = tf.constant(self.sample_length, dtype=tf.int32, shape=[self.number_of_crops])
        limit = shapes - sizes + 1
        offset = tf.random_uniform(tf.shape(shapes), dtype=sizes.dtype, maxval=sizes.dtype.max, seed=3) % limit

        # Crop files tensor according to the pre-computed indices
        cropped_files = tf.map_fn(lambda o: tf.slice(files, tf.convert_to_tensor([o]), [self.sample_length]), offset,
                                  dtype=files.dtype)

        return tf.data.Dataset.from_tensor_slices(cropped_files)

    def _map_func_prepare_sequences(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to prepare parsed and batched sequences. Reshapes along temporal axis and encodes labels."""

        # Convert labels into one-hot-encoded tensors
        action_encoded = tf.one_hot(
            tf.squeeze(tf.gather(action, tf.range(self.sample_length, delta=self.sample_length))), _NUM_ACTIONS)
        location_encoded = tf.one_hot(
            tf.squeeze(tf.gather(location, tf.range(self.sample_length, delta=self.sample_length))), _NUM_LOCATIONS)
        subject_encoded = tf.one_hot(
            tf.squeeze(tf.gather(subject, tf.range(self.sample_length, delta=self.sample_length))), _NUM_SUBJECTS)

        # Reshape audio_images to be the length of a full video of sample_length seconds
        if self.include_audio_images:
            reshaped_audio_images = tf.reshape(audio_images, [-1, 36, 48, _NUMBER_OF_CHANNELS])
        else:
            reshaped_audio_images = tf.zeros([], tf.int32)

        # Reshape audio_samples to be the length of a full video of sample_length seconds
        if self.include_audio_data:
            reshaped_audio_samples = tf.reshape(audio_samples, [-1, _NUMBER_OF_TRACKS, _NUMBER_OF_SAMPLES])
        else:
            reshaped_audio_samples = tf.zeros([], tf.int32)

        # Reshape audio_samples to be the length of a full video of sample_length seconds
        if self.include_video_data:
            reshaped_video_images = tf.reshape(video_images, [-1, 480, 640, 3])
        else:
            reshaped_video_images = tf.zeros([], tf.int32)

        return reshaped_audio_images, reshaped_audio_samples, reshaped_video_images, action_encoded, location_encoded, subject_encoded

    def _map_func_audio_samples_build_wav(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to build waveform audio from raw audio samples."""

        audio_wav = tf.py_func(self._build_wav_py_function, [audio_samples], tf.float32)

        return audio_images, audio_wav, video_images, action, location, subject

    def _build_wav_py_function(self, audio_data):
        """Python function to build a waveform audio from audio samples."""

        mic_id = 0

        # Compose audio time series
        audio_data_mic = audio_data[:, mic_id, :].astype(np.float32)
        audio_data_mic_flat = audio_data_mic.flatten('C')
        audio_data_mic_norm = audio_data_mic_flat / abs(
            max(audio_data_mic_flat.min(), audio_data_mic_flat.max(), key=abs))

        # Re-sample audio to 22 kHz
        audio_wav = librosa.core.resample(audio_data_mic_norm, audio_data_mic_norm.shape[0] / self.sample_length,
                                          self.sample_rate)

        # Make range [-256, 256]
        audio_wav = audio_wav / abs(max(audio_wav.min(), audio_wav.max(), key=abs))
        audio_wav *= 256.0

        return audio_wav

    def _map_func_audio_samples_normalize_spectrograms(self, audio_images, audio_samples, video_images,
                                                       action, location, subject):
        """Input mapper to apply min-max normalization to the audio samples."""

        audio_samples_norm = tf.divide(tf.subtract(audio_samples, self.global_min),
                                       tf.subtract(self.global_max, self.global_min))

        return audio_images, audio_samples_norm, video_images, action, location, subject

    def _map_func_video_images_pick_frames(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to pick randomly five frames."""

        selected_video_images = self._pick_random_frames(video_images)

        return audio_images, audio_samples, selected_video_images, action, location, subject

    def _pick_random_frames(self, video_images):
        num_frames = tf.shape(video_images)[0]  # how many images
        n_to_sample = tf.constant([_NUMBER_OF_FRAMES])  # 5
        mask = self._sample_mask(num_frames, n_to_sample)  # pick in all 5 if is_training
        frames = tf.boolean_mask(video_images, mask)  # keep element in ones position
        return frames

    def _sample_mask(self, num_frames, sample_size):
        # randomly choose between uniform or random sampling
        end = tf.subtract(num_frames, 1)
        indexes = tf.to_int32(tf.linspace(
            0.0, tf.to_float(end), sample_size[0]))  # uses linspace to draw 5 samples in all_samples
        # find indexes among 60
        updates = tf.ones(sample_size, dtype=tf.int32)  # one in 5 positions
        mask = tf.scatter_nd(tf.expand_dims(indexes, 1),
                             updates, tf.expand_dims(num_frames, 0))

        compare = tf.ones([num_frames], dtype=tf.int32)
        mask = tf.equal(mask, compare)
        return mask

    def _map_func_video_images(self, audio_images, audio_samples, video_images, action, location, subject):
        """Input mapper to pre-processes the given images."""

        def prepare_image(image):
            return vgg_preprocessing.preprocess_image(image, _IMAGE_SIZE, _IMAGE_SIZE, is_training=False)

        processed_images = tf.map_fn(prepare_image, video_images, dtype=tf.float32, back_prop=False)

        return audio_images, audio_samples, processed_images, action, location, subject

    def _map_func_audio_samples_build_spectrogram(self, audio_images, audio_wav, processed_images, action, location,
                                                  subject):
        """Input mapper to build spectrogram from waveform audio."""

        audio_stfts = tf.contrib.signal.stft(audio_wav, frame_length=self.frame_length,
                                             frame_step=self.frame_step, fft_length=self.fft_length)

        magnitude_spectrograms = tf.expand_dims(tf.abs(audio_stfts), 1)

        return audio_images, magnitude_spectrograms, processed_images, action, location, subject

    @property
    def total_batches(self):
        total_batches = int(math.ceil(self.num_samples / self.batch_size))
        return total_batches
