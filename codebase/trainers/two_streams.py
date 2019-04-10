from datetime import datetime
from codebase.trainers import build_accuracy

import tensorflow as tf

flags = tf.app.flags
slim = tf.contrib.slim
FLAGS = flags.FLAGS


class TwoStreamsTrainer(object):

    def __init__(self, teacher_model, student_model, logger=None,
                 lambda_value=0.5, temperature_value=1.0,
                 display_freq=1, learning_rate=0.0001, num_classes=14, num_epochs=1):

        self.teacher_model = teacher_model
        self.student_model = student_model
        self.logger = logger

        self.display_freq = display_freq
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs

        output_shape = self.teacher_model.output.get_shape().as_list()[1:]
        output_shape = [-1, 12 * FLAGS.sample_length] + output_shape

        teacher_logits_output = tf.reduce_mean(tf.reshape(self.teacher_model.output, shape=output_shape), axis=1)
        student_logits_output = self.student_model.output

        self.student_labels = tf.placeholder(tf.float32, [None, 14], 'labels')
        teacher_labels = tf.nn.softmax(teacher_logits_output / temperature_value)

        # Extract input model shape
        self.teacher_shape = self.teacher_model.network['input'].get_shape().as_list()
        self.student_shape = self.student_model.network['input'].get_shape().as_list()

        # Define losses
        self.cross_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=self.student_labels,
            logits=student_logits_output,
            weights=1.0 - lambda_value,
            scope='cross_loss'
        )
        self.dist_loss = tf.losses.softmax_cross_entropy(
            onehot_labels=teacher_labels,
            logits=student_logits_output,
            weights=lambda_value,
            scope='dist_loss'
        )
        self.loss = tf.losses.get_total_loss()

        # Define accuracy
        self.accuracy = build_accuracy(student_logits_output, self.student_labels)

        # Initialize counters and stats
        self.global_step = tf.train.create_global_step()

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = slim.learning.create_train_op(total_loss=self.loss,
                                                        optimizer=self.optimizer,
                                                        global_step=self.global_step,
                                                        variables_to_train=self.student_model.train_vars)

        # Initialize model saver
        self.saver = tf.train.Saver(max_to_keep=None)

    def _get_optimizer_variables(self, optimizer):

        optimizer_vars = [optimizer.get_slot(var, name) for name in optimizer.get_slot_names() for var in
                          self.student_model.train_vars if var is not None]

        optimizer_vars.extend(list(optimizer._get_beta_accumulators()))

        return optimizer_vars

    def _init_model(self, session):

        if FLAGS.restore_checkpoint is None:
            # Initialize global step
            print('{}: {} - Initializing global step'.format(datetime.now(), FLAGS.exp_name))
            session.run(self.global_step.initializer)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            # Initialize optimizer variables
            print('{}: {} - Initializing optimizer variables'.format(datetime.now(), FLAGS.exp_name))
            optimizer_vars = self._get_optimizer_variables(self.optimizer)
            optimizer_init_op = tf.variables_initializer(optimizer_vars)
            session.run(optimizer_init_op)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            # Initialize teacher model
            print('{}: {} - Initializing teacher model'.format(datetime.now(), FLAGS.exp_name))
            saver = tf.train.Saver(var_list=slim.get_model_variables(self.teacher_model.scope))
            saver.restore(session, FLAGS.teacher_restore_checkpoint)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

            # Initialize student model
            print('{}: {} - Initializing student model'.format(datetime.now(), FLAGS.exp_name))
            if FLAGS.student_init_checkpoint is not None:
                self.student_model.init_model(session, FLAGS.student_init_checkpoint)
            else:
                init_op = tf.variables_initializer(slim.get_model_variables(self.student_model.scope))
                session.run(init_op)
            print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))
        else:
            # Restore session from checkpoint
            self._restore_model(session)

    def _restore_model(self, session):

        # Restore model
        print('{}: {} - Restoring session'.format(datetime.now(), FLAGS.exp_name))
        saver = tf.train.Saver()
        saver.restore(session, FLAGS.restore_checkpoint)
        print('{}: {} - Done'.format(datetime.now(), FLAGS.exp_name))

    def train(self, train_data=None, valid_data=None):

        # Assert training and validation sets are not None
        assert train_data is not None
        assert valid_data is not None

        # Add trainable variables to the summary
        # for var in self.student_model.train_vars:
        #     self.logger.log_histogram(var.name, var)

        # Add losses to summary
        self.logger.log_scalar('cross_entropy_loss', self.cross_loss)
        self.logger.log_scalar('distillation_loss', self.dist_loss)
        self.logger.log_scalar('train_loss', self.loss)

        # Add accuracy to the summary
        self.logger.log_scalar('train_accuracy', self.accuracy)

        # Merge all summaries together
        self.logger.merge_summary()

        # Create a re-initializable iterator given the dataset structure
        iterator = tf.data.Iterator.from_structure(train_data.data.output_types,
                                                   train_data.data.output_shapes)
        next_batch = iterator.get_next()

        # Create operation for initializing the iterator
        training_init_op = iterator.make_initializer(train_data.data)

        # Start training session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as session:

            # Initialize model either randomly or with a checkpoint
            self._init_model(session)

            # Add the model graph to TensorBoard
            self.logger.write_graph(session.graph)

            start_epoch = int(tf.train.global_step(session, self.global_step) / train_data.total_batches)
            best_epoch = -1
            best_accuracy = -1.0
            best_loss = -1.0

            # For each epoch
            for epoch in range(start_epoch, start_epoch + self.num_epochs):

                # Initialize counters and stats
                step = 0

                # Initialize iterator over the training set
                session.run(training_init_op, feed_dict={train_data.seed: epoch})

                # For each mini-batch
                while True:
                    try:

                        # Prepare batch
                        teacher_data, student_data, labels_data = session.run([
                            tf.reshape(next_batch[0], shape=[-1, self.teacher_shape[1], self.teacher_shape[2], self.teacher_shape[3]]),
                            tf.reshape(next_batch[1], shape=[-1, self.student_shape[1], self.student_shape[2], self.student_shape[3]]),
                            tf.reshape(next_batch[3], shape=[-1, self.num_classes])
                        ])

                        # Compute mini-batch error
                        if step % self.display_freq == 0:
                            train_loss, train_accuracy, train_summary = session.run(
                                [self.loss, self.accuracy, self.logger.summary_op],
                                feed_dict={
                                    self.teacher_model.network['input']: teacher_data,
                                    self.student_model.network['input']: student_data,
                                    self.student_labels: labels_data,
                                    self.teacher_model.network['keep_prob']: 1.0,
                                    self.student_model.network['keep_prob']: 1.0,
                                    self.teacher_model.network['is_training']: False,
                                    self.student_model.network['is_training']: False})

                            print('{}: {} - Iteration: [{:3}]\t Training_Loss: {:6f}\t Training_Accuracy: {:6f}'.format(
                                datetime.now(), FLAGS.exp_name, step, train_loss, train_accuracy))

                            self.logger.write_summary(train_summary, tf.train.global_step(session, self.global_step))

                        # Forward batch through the network
                        session.run(self.train_step, feed_dict={self.teacher_model.network['input']: teacher_data,
                                                                self.student_model.network['input']: student_data,
                                                                self.student_labels: labels_data,
                                                                self.teacher_model.network['keep_prob']: 1.0,
                                                                self.student_model.network['keep_prob']: 0.5,
                                                                self.teacher_model.network['is_training']: False,
                                                                self.student_model.network['is_training']: True})

                        # Update counters and stats
                        step += 1

                    except tf.errors.OutOfRangeError:
                        break

                # Save model
                if FLAGS.save_best_only is False:
                    self._save_checkpoint(session, epoch)

                # Evaluate model on validation set
                total_loss, total_accuracy = self._valid(session, valid_data)

                print('{}: {} - Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                        FLAGS.exp_name,
                                                                                                        epoch,
                                                                                                        total_loss,
                                                                                                        total_accuracy))

                self.logger.write_summary(tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss", simple_value=total_loss),
                    tf.Summary.Value(tag="valid_accuracy", simple_value=total_accuracy)
                ]), epoch)

                self.logger.flush_writer()

                if total_accuracy >= best_accuracy:
                    best_epoch = epoch
                    best_accuracy = total_accuracy
                    best_loss = total_loss

                    self._save_checkpoint(session)

            print('{}: {} - Best Epoch: {}\t Validation_Loss: {:6f}\t Validation_Accuracy: {:6f}'.format(datetime.now(),
                                                                                                         FLAGS.exp_name,
                                                                                                         best_epoch,
                                                                                                         best_loss,
                                                                                                         best_accuracy))

    def _save_checkpoint(self, session, epoch=None):

        checkpoint_dir = '{}/{}'.format(FLAGS.checkpoint_dir, FLAGS.exp_name)
        model_name = 'model.ckpt' if epoch is None else 'epoch_{}.ckpt'.format(epoch)

        if not tf.gfile.Exists(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)

        print('{}: {} - Saving model to {}/{}'.format(datetime.now(), FLAGS.exp_name, checkpoint_dir, model_name))

        self.saver.save(session, '{}/{}'.format(checkpoint_dir, model_name))

    def _valid(self, session, valid_data):

        # Create a one-shot iterator
        iterator = valid_data.data.make_one_shot_iterator()
        next_batch = iterator.get_next()

        return self._evaluate(session, next_batch)

    def _evaluate(self, session, next_batch):

        # Initialize counters and stats
        loss_sum = 0
        accuracy_sum = 0
        data_set_size = 0

        # For each mini-batch
        while True:
            try:

                # Prepare batch
                teacher_data, student_data, labels_data = session.run([
                    tf.reshape(next_batch[0], shape=[-1, self.teacher_shape[1], self.teacher_shape[2], self.teacher_shape[3]]),
                    tf.reshape(next_batch[1], shape=[-1, self.student_shape[1], self.student_shape[2], self.student_shape[3]]),
                    tf.reshape(next_batch[3], shape=[-1, 14])
                ])

                # Compute batch loss
                batch_loss, batch_accuracy = session.run([self.loss, self.accuracy],
                                                         feed_dict={self.teacher_model.network['input']: teacher_data,
                                                                    self.student_model.network['input']: student_data,
                                                                    self.student_labels: labels_data,
                                                                    self.teacher_model.network['keep_prob']: 1.0,
                                                                    self.student_model.network['keep_prob']: 1.0,
                                                                    self.teacher_model.network['is_training']: False,
                                                                    self.student_model.network['is_training']: False})

                # Update counters
                data_set_size += labels_data.shape[0]
                loss_sum += batch_loss * labels_data.shape[0]
                accuracy_sum += batch_accuracy * labels_data.shape[0]

            except tf.errors.OutOfRangeError:
                break

        total_loss = loss_sum / data_set_size
        total_accuracy = accuracy_sum / data_set_size

        return total_loss, total_accuracy

    def test(self, test_data=None):

        # Assert testing set is not None
        assert test_data is not None

        # Create a one-shot iterator
        iterator = test_data.data.make_one_shot_iterator()
        next_batch = iterator.get_next()

        # Start training session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:

            # Initialize model either randomly or with a checkpoint if given
            self._restore_model(session)

            # Evaluate model over the testing set
            test_loss, test_accuracy = self._evaluate(session, next_batch)

        print('{}: {} - Testing_Loss: {:6f}\t Testing_Accuracy: {:6f}'.format(datetime.now(),
                                                                              FLAGS.exp_name,
                                                                              test_loss,
                                                                              test_accuracy))

        return test_loss, test_accuracy
