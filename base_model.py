# -*- coding: utf-8 -*-
import tensorflow as tf
from datetime import datetime
import os
import logging
import numpy as np
import pprint as pp
import os
import time

# 设置gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True
gpuConfig=None


class baseNN(object):

    def __init__(self,
                 reader,
                 batch_size=128,
                 num_training_steps=20000,
                 learning_rate=.01,
                 optimizer='adam',
                 grad_clip=1,
                 clip=True,
                 regularization_constant=0.0,
                 keep_prob=1.0,
                 early_stopping_steps=3000,
                 warm_start_init_step=0,
                 num_restarts=None,
                 enable_parameter_averaging=False,
                 min_steps_to_checkpoint=100,
                 log_interval=20,
                 loss_averaging_window=100,
                 num_validation_batches=4397,
                 log_dir='logs',
                 checkpoint_dir='checkpoints',
                 predict_dir='predictions'):
        """
        init all the variables and generate the graph
        :param reader: dataset reader
        :param batch_size: batch size
        :param num_training_steps:  train step
        :param learning_rate: learning rate 1e-5
        :param optimizer: 'adam'
        :param grad_clip: gradient clip 1
        :param clip: bool
        :param regularization_constant: no use
        :param keep_prob: 0.5
        :param early_stopping_steps: 100
        :param warm_start_init_step: no use
        :param num_restarts: no use
        :param enable_parameter_averaging: no use
        :param min_steps_to_checkpoint:
        :param log_interval: logging interval
        :param loss_averaging_window: no use
        :param num_validation_batches: all the batches
        :param log_dir: $base_dir/log
        :param checkpoint_dir:  $base_dir/checkpoint_dir
        :param predict_dir: $base_dir/predict_dir/
        """
        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.clip = clip
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.keep_prob_scalar = keep_prob
        self.early_stopping_steps = early_stopping_steps
        self.warm_start_init_step = warm_start_init_step
        self.num_restarts = num_restarts
        self.enable_parameter_averaging = enable_parameter_averaging
        self.min_steps_to_checkpoint = min_steps_to_checkpoint
        self.log_interval = log_interval
        self.loss_averaging_window = loss_averaging_window
        self.num_validation_batches = num_validation_batches

        # makedirs
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.predict_dir = predict_dir
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        self.init_logging(self.log_dir)
        #logging.info('\new run with parameters:\n{}'.format(pp.pformat(self.__dict__)))
        self.graph = self.build_graph()
        self.session = tf.Session(config=gpuConfig, graph=self.graph)

    def train(self):
        """
        train the wavenet model
        :return: none
        """

        with self.session.as_default():
            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                step = self.warm_start_init_step
            else:
                self.session.run(self.init)
                step = 0
            init = tf.global_variables_initializer()
            init.run()
            train_generator = self.reader.train_batch_generator(self.batch_size)
            val_generator = self.reader.val_batch_generator(self.num_validation_batches * self.batch_size).next()
            val_feed_dict = {
                getattr(self, placeholder_name, None): val_generator[placeholder_name]
                for placeholder_name in val_generator if hasattr(self, placeholder_name)
            }
            if hasattr(self, 'keep_prob'):
                val_feed_dict.update({self.keep_prob: 1.0})
            best_validation_loss, self.best_validation_tstep = float('inf'), 0
            start = time.time()
            while step < self.num_training_steps:
                batch_data = train_generator.next()
                train_feed_dict = {
                    getattr(self, placeholder_name, None): batch_data[placeholder_name]
                    for placeholder_name in batch_data if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                train_loss, _ = self.session.run(
                    fetches=[self.loss, self.step],
                    feed_dict=train_feed_dict
                )

                if step != 0 and step % self.log_interval == 0:
                    second = time.time()
                    logging.info("100 batch train time:{}".format(second - start))
                    start = second
                    tf.local_variables_initializer().run()
                    val_test_loss, val_train_loss = self.session.run(
                        [self.mae, self.loss],
                        feed_dict=val_feed_dict
                    )
                    metric_log = (
                        "[[step {:>8}]]     "
                        "[[train]]     loss: {:<12}     "
                        "[[val_test]]     loss: {:<12}     "
                        "[[val_train]]      loss: {:<12}       "
                    ).format(step, round(train_loss, 8), round(val_test_loss, 8), round(val_train_loss, 8))
                    logging.info(metric_log)
                    if val_test_loss < best_validation_loss:
                        self.best_validation_tstep = step
                        best_validation_loss = val_test_loss
                        if self.best_validation_tstep > self.min_steps_to_checkpoint:
                            self.save(self.best_validation_tstep)

                step += 1
            self.save(step)

    def restore(self, step=None, averaged=False):
        """
        restore the model
        :param step: train step
        :param averaged: none
        :return:  none
        """
        saver = self.saver
        checkpoint_dir = self.checkpoint_dir
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model-{}'.format(step)
            )
            logging.info('restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)


    def save(self, step, averaged=False):
        """
        save the model
        :param step: train step
        :param averaged: none
        :return: none
        """
        saver = self.saver
        checkpoint_dir = self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}-{}'.format(model_path, step))
        saver.save(self.session, model_path, global_step=step)

    def predict_incremental(self,waveform):
        raise NotImplementedError('subclass must implement this')

    def predict_incre(self):
        """
        predict using fast wavenet algorithm
        :return:
        """
        waveform = tf.placeholder(tf.float32, [None, None,2*self.feature_dim])
        proba_fast = self.predict_incremental(waveform)
        test_generator = self.reader.test_batch_generator(self.num_validation_batches * self.batch_size).next()
        data = test_generator['data']
        mean = test_generator['d7mean']
        label = test_generator['label']
        test_data = np.concatenate([data,mean],axis=-1)

        self.session.run(tf.global_variables_initializer())
        self.session.run(self.init_ops)
        # Prime the incremental generation with all samples
        # except the last one
        predict = []
        next_predict = None
        for i in range(self.seq_len):
            test = np.expand_dims(test_data[:,i,:],axis=1)
            proba_fast_predict, _ = self.session.run(
                    [proba_fast, self.push_ops],
                    feed_dict={waveform: test,self.keep_prob:1.0})
            if i == self.seq_len - 1:
                next_predict = proba_fast_predict
                predict.append(np.expand_dims(next_predict, -1))

        for i in range(self.num_step-1):
            other_feature = test_data[:,self.seq_len+i,1:]
            if len(other_feature.shape)<3:
                other_feature = np.expand_dims(other_feature,axis=-1)
            next_predict = np.expand_dims(next_predict, axis=-1)
            next_predict = np.concatenate([next_predict,other_feature],axis=-1)
            data,_ = self.session.run(
                [proba_fast, self.push_ops],
                feed_dict={waveform: next_predict,self.keep_prob:1.0})
            next_predict = data

            # Get the last sample from the incremental generator
            predict.append(np.expand_dims(data, -1))
        predict_out = np.concatenate(predict, 1)
        predict_value = np.expm1(predict_out+mean[:,-288:,:])
        label = np.expm1(label)
        mae = np.sum(np.abs(predict_value-label))/predict_out.shape[0]/predict_out.shape[1]

        metric_log = ("[[predict test]]     loss: {:<12}     ").format(round(mae, 8))
        logging.info(metric_log)


        predict_name = "predict_out"
        save_file = os.path.join(self.predict_dir,
                                 '{}_{}.npy'.format(predict_name, datetime.now().strftime('%Y-%m-%d_%H-%M')))
        logging.info('saving {} with shape {} to {}'.format(predict_name, predict_out.shape, save_file))
        np.save(save_file, predict_value)

        label_name = "label_name"
        save_file = os.path.join(self.predict_dir,
                                 '{}_{}.npy'.format(label_name, datetime.now().strftime('%Y-%m-%d_%H-%M')))
        logging.info('saving {} with shape {} to {}'.format(label_name, test_generator['label'].shape, save_file))
        np.save(save_file, np.expm1(test_generator['label']))

    def predict(self):
        test_generator = self.reader.test_batch_generator(self.num_validation_batches * self.batch_size).next()
        test_feed_dict = {
            getattr(self, placeholder_name, None): test_generator[placeholder_name]
            for placeholder_name in test_generator if hasattr(self, placeholder_name)
        }
        if hasattr(self, 'keep_prob'):
            test_feed_dict.update({self.keep_prob: 1.0})
        predict_out, mae = self.session.run(
            [self.predict_out, self.mae],
            feed_dict=test_feed_dict
        )
        metric_log = ("[[predict test]]     loss: {:<12}     ").format(round(mae, 8))
        logging.info(metric_log)
        predict_name = "predict_out"
        save_file = os.path.join(self.predict_dir,
                                 '{}_{}.npy'.format(predict_name, datetime.now().strftime('%Y-%m-%d_%H-%M')))
        logging.info('saving {} with shape {} to {}'.format(predict_name, predict_out.shape, save_file))
        np.save(save_file, predict_out)

        label_name = "label_name"
        save_file = os.path.join(self.predict_dir,
                                 '{}_{}.npy'.format(label_name, datetime.now().strftime('%Y-%m-%d_%H-%M')))
        logging.info('saving {} with shape {} to {}'.format(label_name, test_generator['label'].shape, save_file))
        np.save(save_file, np.expm1(test_generator['label']))


    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')

    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())

    def update_parameters(self, loss, optimizer,clip=False):

        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([
                tf.sqrt(tf.reduce_sum(tf.square(tf.cast(param,tf.float32))) for param in tf.trainable_variables())])
            #l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            l2_norm = tf.Print(l2_norm,['l2_norm:',l2_norm])
            #l2_norm = tf.Print(l2_norm,["l2_norm:",l2_norm])
            # l2_norm = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # l2_norm = sum(l2_norm)
            # l2_norm = tf.cast(l2_norm, tf.float32)
            # l2_norm = tf.Print(l2_norm,["l2_norm:",l2_norm])
            loss = loss + self.regularization_constant*l2_norm
        if clip:
            # compute the gradient
            grads = optimizer.compute_gradients(loss)

            # clip the value
            grads = [ (tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) if g is not None else (g,v_) for g,v_ in grads]
            #grads = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in grads]

            step = optimizer.apply_gradients(grads, global_step=self.global_step)
            #grads_vars = [v for (g, v) in grads if g is not None]  # all variable that has gradients
            #gradient = optimizer.compute_gradients(loss, grads_vars)  # gradient of network (without NoneType)

            #grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
            #               for (g, v) in gradient]
            #grads = [(tf.clip_by_value(g, -self.grad_clip, self.grad_clip), v_) for g, v_ in gradient]
            #grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
            #                for (g, v) in gradient]
        else:
            step = optimizer.minimize(loss)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        # logging.info('all parameters:')
        #logging.info(pp.pformat([(var.name, var.shape) for var in tf.global_variables()]))

        logging.info('trainable parameters:')
        logging.info(pp.pformat([(var.name, var.shape) for var in tf.trainable_variables()]))

        # logging.info('trainable parameter count:')
        # logging.info(str(np.sum(np.prod(np.array(var.shape.as_list())) for var in tf.trainable_variables())))

    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'

    def build_graph(self):
        default_graph = tf.get_default_graph()
        with default_graph.as_default() as graph:
            # self.ema = tf.train.ExponentialMovingAverage(decay=0.995)
            self.global_step = tf.Variable(0)
            self.learning_rate_var = tf.train.exponential_decay(
                self.learning_rate,
                self.global_step,
                decay_steps=1000,
                decay_rate=0.96,
                staircase=True)
            self.loss, self.mae = self.calculate_loss()
            self.optimizer = self.get_optimizer(self.learning_rate_var)
            self.update_parameters(self.loss, self.optimizer,clip=self.clip)

            self.saver = tf.train.Saver(max_to_keep=5)
            # if self.enable_parameter_averaging:
            #     self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()

            return graph

