import tempfile
import tensorflow as tf


class Logger(object):

    def __init__(self, log_dir):

        self.__log_dir = tempfile.mkdtemp() if log_dir is None or not tf.gfile.Exists(log_dir) else log_dir
        self.summary_op = None
        self.__summary_inputs = []
        self.__file_writer = tf.summary.FileWriter(self.__log_dir)

    def log_scalar(self, tag, value):
        self.__summary_inputs.append(tf.summary.scalar(tag, value))

    def log_histogram(self, tag, value):
        self.__summary_inputs.append(tf.summary.histogram(tag, value))

    def log_image(self, tag, value, max_outputs=3):
        self.__summary_inputs.append(tf.summary.image(tag, value, max_outputs=max_outputs))

    def log_sound(self, tag, value, sample_rate=22050, max_outputs=3):
        self.__summary_inputs.append(tf.summary.audio(tag, value, sample_rate, max_outputs=max_outputs))

    def write_graph(self, graph, global_step=None):
        self.__file_writer.add_graph(graph, global_step)

    def write_summary(self, train_summary, global_step=None):
        self.__file_writer.add_summary(train_summary, global_step)

    def merge_summary(self):
        self.summary_op = tf.summary.merge(self.__summary_inputs)

    def flush_writer(self):
        self.__file_writer.flush()
