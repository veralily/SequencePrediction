from __future__ import print_function
import os
import sys
import time
import random
import argparse
import tensorflow as tf
import numpy as np
import utils
import read_data
import model_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def print_in_file(sstr):
    sys.stdout.write(str(sstr) + '\n')
    sys.stdout.flush()
    os.fsync(sys.stdout)


def make_noise(shape):
    return tf.random_normal(shape)


class T_Pred(object):
    def __init__(self, config, cell_type, event_file, time_file, is_training):
        self.alpha = 1.0
        self.cell_type = cell_type
        self.event_file = event_file
        self.time_file = time_file
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.g_size = config.g_size
        self.filter_output_dim = config.filter_output_dim
        self.filter_size = config.filter_size
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.n_g = config.num_gen
        self.is_training = is_training
        self.keep_prob = config.keep_prob
        self.res_rate = config.res_rate
        self.length = 1
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.lr = config.learning_rate
        self.LAMBDA = config.LAMBDA
<<<<<<< HEAD
        self.gamma = config.gamma
        self.train_data, self.valid_data, self.test_data = read_data.data_split(
            event_file, time_file, shuffle=False)

=======
        self.delta = config.delta
        self.gamma = config.gamma
        self.event_to_id = read_data.build_vocab(self.event_file)
        self.train_data, self.valid_data, self.test_data = read_data.data_split(
            event_file, time_file, shuffle=False)
        self.embeddings = tf.get_variable(
            "embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
        self.sample_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps + self.length])
        self.target_t = tf.placeholder(tf.float32, [self.batch_size, self.length])
        self.inputs_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
        self.targets_e = tf.placeholder(tf.int64, [self.batch_size, self.length])
        self.input_e = tf.placeholder(tf.int64, [self.batch_size, self.num_steps])
        self.build()

    def encoder_e(self, cell_type, inputs):
        with tf.variable_scope('Generator'):
            outputs_e = utils.build_encoder_graph_gru(
                inputs,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                self.num_steps,
                self.keep_prob,
                self.is_training,
                "Encoder_e" + cell_type)
            hidden_re = [tf.expand_dims(output_e, 1) for output_e in outputs_e]
            hidden_re = tf.concat(hidden_re, 1)
            return hidden_re

    def g_event(self, hidden_r, name=''):
        """
        The generative model for time and event
        mode:
        1. use the concatenated hidden representation for each time step
        2. use the unfolded hidden representation separately for each time step
        """

        with tf.variable_scope("Generator_E" + name):
            outputs = utils.build_rnn_graph_decoder1(
                hidden_r,
                self.num_layers,
                self.g_size,
                self.batch_size,
                self.length,
                "G_E.RNN")
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.g_size])
            output = utils.linear('G_E.Output', self.g_size, self.vocab_size, output)
            logits = tf.reshape(output, [self.batch_size, self.length, self.vocab_size])
            return logits

    def params_with_name(self, name):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [v for v in variables if name in v.name]

<<<<<<< HEAD
    def params_all(self):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return variables

=======
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
    def loss(self, pred_e, real_e):

        gen_e_cost = tf.contrib.seq2seq.sequence_loss(pred_e, real_e, weights=tf.ones([self.batch_size, self.length]),
                                                      name="SeqLoss")
        '''The separate training of Generator and Discriminator'''
<<<<<<< HEAD
        gen_params = self.params_all()

        '''Use the Adam Optimizer to update the variables'''
        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_e_cost, var_list=gen_params)

        '''Use the RMSProp Optimizer to update the variables'''
        # gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(gen_e_cost, var_list=gen_params)
=======
        gen_params = self.params_with_name('Generator')

        '''Use the Adam Optimizer to update the variables'''
        # gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        # disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

        '''Use the RMSProp Optimizer to update the variables'''
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(gen_e_cost, var_list=gen_params)
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00

        return gen_train_op, gen_e_cost

    def build(self):
        """
        build the model
        define the loss function
        define the optimization method
        """
<<<<<<< HEAD
        embeddings = tf.get_variable(
            "embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
        inputs_e = tf.nn.embedding_lookup(embeddings, self.input_e)

        hidden_re = self.encoder_e(self.cell_type, inputs_e)

        pred_e = self.g_event(tf.reshape(hidden_re, [self.batch_size, -1]))
=======
        self.inputs_e = tf.nn.embedding_lookup(self.embeddings, self.input_e)

        hidden_re = self.encoder_e(self.cell_type, self.inputs_e)

        self.pred_e = self.g_event(tf.reshape(hidden_re, [self.batch_size, -1]))
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
        # self.pred_e = self.g_event(output_re)
        # use the extracted feature from input events and timestamps to generate the time sequence

        gen_train_op, gen_e_cost = self.loss(
<<<<<<< HEAD
            pred_e,
=======
            self.pred_e,
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
            self.targets_e)

        self.g_train_op = gen_train_op
        self.gen_e_cost = gen_e_cost
<<<<<<< HEAD
        self.pred_e = pred_e

        print('pred_e shape: ')
        print(pred_e.get_shape())
=======

        print('pred_e shape: ')
        print(self.pred_e.get_shape())
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
        print('targets_e shape: ')
        print(self.targets_e.get_shape())

        # MRR@k
        self.batch_precision, self.batch_precision_op = tf.metrics.average_precision_at_k(
<<<<<<< HEAD
            labels=self.targets_e, predictions=pred_e, k=20, name='precision_k')
=======
            labels=self.targets_e, predictions=self.pred_e, k=20, name='precision_k')
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
        # Isolate the variables stored behind the scenes by the metric operation
        self.running_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_k")
        # Define initializer to initialize/reset running variables
        self.running_precision_vars_initializer = tf.variables_initializer(var_list=self.running_precision_vars)

        # Recall@k
        self.batch_recall, self.batch_recall_op = tf.metrics.recall_at_k(
<<<<<<< HEAD
            labels=self.targets_e, predictions=pred_e, k=20, name='recall_k')
=======
            labels=self.targets_e, predictions=self.pred_e, k=20, name='recall_k')
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
        # Isolate the variables stored behind the scenes by the metric operation
        self.running_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_k")
        # Define initializer to initialize/reset running variables
        self.running_recall_vars_initializer = tf.variables_initializer(var_list=self.running_recall_vars)

        self.saver = tf.train.Saver(max_to_keep=None)

    def train(self, sess, args):
        self.logdir = args.logdir + parse_time()
        while os.path.exists(self.logdir):
            time.sleep(random.randint(1, 5))
            self.logdir = args.logdir + parse_time()
        os.makedirs(self.logdir)

        if not os.path.exists('%s/logs' % self.logdir):
            os.makedirs('%s/logs' % self.logdir)

        if args.weights is not None:
            self.saver.restore(sess, args.weights)

        self.lr = self.learning_rate

        for epoch in range(args.iters):
            '''training'''

<<<<<<< HEAD
            if epoch > 0 and epoch % (args.iters // 5) == 0:
                self.lr = self.lr * 2. / 3

=======
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
            sess.run([self.running_precision_vars_initializer, self.running_recall_vars_initializer])
            # re initialize the metric variables of metric.precision and metric.recall,
            # to calculate these metric for each epoch
            batch_precision, batch_recall = 0.0, 0.0

            sum_iter = 0.0

<<<<<<< HEAD
            input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
                self.train_data,
                self.num_steps,
                self.length)

            sample_t = read_data.generate_sample_t(
                self.batch_size,
                input_time_data,
                target_time_data)

            i = 0

            for e_x, e_y, t_x, t_y in read_data.generate_batch(
                    self.batch_size,
                    input_event_data,
                    target_event_data,
                    input_time_data,
                    target_time_data):

                feed_dict = {
                    self.input_e: e_x,
                    self.inputs_t: np.maximum(np.log(t_x), 0),
                    self.target_t: t_y,
                    self.targets_e: e_y,
                    self.sample_t: np.maximum(np.log(sample_t), 0)}
=======
            input_len, input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
                self.train_data,
                self.event_to_id,
                self.num_steps,
                self.length)

            batch_num, e_x_list, e_y_list, t_x_list, t_y_list = read_data.generate_batch(
                input_len,
                self.batch_size,
                input_event_data,
                target_event_data,
                input_time_data,
                target_time_data)

            _, sample_t_list = read_data.generate_sample_t(
                input_len,
                self.batch_size,
                input_time_data,
                target_time_data)

            g_iters = 5
            gap = g_iters + 1

            for i in range(batch_num):

                if i > 0 and i % (batch_num // 10) == 0:
                    self.lr = self.lr * 2. / 3

                feed_dict = {
                    self.input_e: e_x_list[i],
                    self.inputs_t: np.maximum(np.log(t_x_list[i]), 0),
                    self.target_t: t_y_list[i],
                    self.targets_e: e_y_list[i],
                    self.sample_t: np.maximum(np.log(sample_t_list[i]), 0)}
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00

                _, _, _, gen_e_cost, batch_precision, batch_recall = sess.run(
                        [self.g_train_op,
                         self.batch_precision_op,
                         self.batch_recall_op,
                         self.gen_e_cost,
                         self.batch_precision,
                         self.batch_recall], feed_dict=feed_dict)
                sum_iter = sum_iter + 1
                # if self.cell_type == 'T_LSTMCell':
                #     sess.run(self.clip_op)

<<<<<<< HEAD
                if i % 100 == 0:
                    print('[epoch: %d, %d] precision: %f, recall: %f, gen_e_loss: %f' % (
                        epoch, int(i // 1000), batch_precision, batch_recall, gen_e_cost))
                    # print('Precision Vars:', sess.run(self.running_precision_vars))
                i += 1
=======
                if i % (batch_num // 10) == 0:
                    print('[epoch: %d, %d] precision: %f, recall: %f, gen_e_loss: %f' % (
                        epoch, int(i // (batch_num // 10)), batch_precision, batch_recall, gen_e_cost))
                    print('Precision Vars:', sess.run(self.running_precision_vars))
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00

            '''evaluation'''

            sess.run([self.running_precision_vars_initializer, self.running_recall_vars_initializer])
            # re initialize the metric variables of metric.precision and metric.recall,
            # to calculate these metric for each epoch

            input_len, input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
                self.valid_data,
<<<<<<< HEAD
                self.num_steps,
                self.length)

            sample_t = read_data.generate_sample_t(
=======
                self.event_to_id,
                self.num_steps,
                self.length)

            batch_num, e_x_list, e_y_list, t_x_list, t_y_list = read_data.generate_batch(
                input_len,
                self.batch_size,
                input_event_data,
                target_event_data,
                input_time_data,
                target_time_data)

            _, sample_t_list = read_data.generate_sample_t(
                input_len,
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
                self.batch_size,
                input_time_data,
                target_time_data)

            sum_iter = 0.0
<<<<<<< HEAD
            i = 0

            self.lr = self.learning_rate

            for e_x, e_y, t_x, t_y in read_data.generate_batch(
                    self.batch_size,
                    input_event_data,
                    target_event_data,
                    input_time_data,
                    target_time_data):

                feed_dict = {
                    self.input_e: e_x,
                    self.inputs_t: np.maximum(np.log(t_x), 0),
                    self.target_t: t_y,
                    self.targets_e: e_y,
                    self.sample_t: np.maximum(np.log(sample_t), 0)}
=======

            self.lr = self.learning_rate

            for i in range(batch_num):
                feed_dict = {
                    self.input_e: e_x_list[i],
                    self.inputs_t: np.maximum(np.log(t_x_list[i]), 0),
                    self.target_t: t_y_list[i],
                    self.targets_e: e_y_list[i],
                    self.sample_t: np.maximum(np.log(sample_t_list[i]), 0)}

                if i > 0 and i % (batch_num // 10) == 0:
                    self.lr = self.lr * 2. / 3
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00

                _, _, gen_e_cost, batch_precision, batch_recall = sess.run(
                    [self.batch_precision_op,
                     self.batch_recall_op,
                     self.gen_e_cost,
                     self.batch_precision,
                     self.batch_recall],
                    feed_dict=feed_dict)

                sum_iter = sum_iter + 1
<<<<<<< HEAD
                i += 1

                if i % 100 == 0:
                    print('%d, precision: %f, recall: %f, gen_e_cost: %f' % (
                        int(i // 1000),
=======

                if i % (batch_num // 10) == 0:
                    print('%d, precision: %f, recall: %f, gen_e_cost: %f' % (
                        int(i // (batch_num // 10)),
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
                        batch_precision,
                        batch_recall,
                        gen_e_cost,
                        ))
                    print('Precision Vars:', sess.run(self.running_precision_vars))
        self.save_model(sess, self.logdir, args.iters)

    def eval(self, sess, args):
        if not os.path.exists(args.logdir + '/output'):
            os.makedirs(args.logdir + '/output')

<<<<<<< HEAD
        # if args.eval_only:
        #     self.test_data = read_data.load_test_dataset(self.dataset_file)
=======
        if args.eval_only:
            self.test_data = read_data.load_test_dataset(self.dataset_file)
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00

        if args.weights is not None:
            self.saver.restore(sess, args.weights)
            print_in_file("Saved")

        lr = self.learning_rate

        batch_size = 100

<<<<<<< HEAD
        input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
            self.test_data,
            self.num_steps,
            self.length)

        sample_t = read_data.generate_sample_t(
            batch_size,
            input_time_data,
            target_time_data)

        f = open(os.path.join(args.logdir, "output_e.txt"), 'w+')
        i = 0

        for e_x, e_y, t_x, t_y in read_data.generate_batch(
                self.batch_size,
                input_event_data,
                target_event_data,
                input_time_data,
                target_time_data):

            feed_dict = {
                self.input_e: e_x,
                self.inputs_t: np.maximum(np.log(t_x), 0),
                # self.target_t : t_y_list[i],
                # self.targets_e : e_y_list[i],
                self.sample_t: np.maximum(np.log(sample_t), 0)}

            pred_e = sess.run(self.pred_e, feed_dict=feed_dict)

=======
        input_len, input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
            self.test_data,
            self.event_to_id,
            self.num_steps,
            self.length,
            shuffle=False)

        batch_num, e_x_list, e_y_list, t_x_list, t_y_list = read_data.generate_batch(
            input_len,
            batch_size,
            input_event_data,
            target_event_data,
            input_time_data,
            target_time_data)

        _, sample_t_list = read_data.generate_sample_t(
            input_len,
            batch_size,
            input_time_data,
            target_time_data)

        f = open(os.path.join(args.logdir, "output_e.txt"), 'w+')

        for i in range(batch_num):
            # print(e_y_list[i])
            feed_dict = {
                self.input_e: e_x_list[i],
                self.inputs_t: np.maximum(np.log(t_x_list[i]), 0),
                # self.target_t : t_y_list[i],
                # self.targets_e : e_y_list[i],
                self.sample_t: np.maximum(np.log(sample_t_list[i]), 0)}

            if i > 0 and i % (batch_num // 10) == 0:
                lr = lr * 2. / 3
            # correct_pred, deviation, pred_e, pred_t, d_loss, g_loss, gen_e_cost, gen_t_cost = sess.run(
            # 	[self.correct_pred, self.deviation, self.pred_e, self.pred_t, self.d_cost, self.g_cost,
            # 	self.gen_e_cost, self.gen_t_cost,self.disc_cost_1, self.gradient_penalty],
            # 	feed_dict = feed_dict)
            pred_e = sess.run(self.pred_e, feed_dict=feed_dict)

            # sum_correct_pred = sum_correct_pred + correct_pred
            # sum_iter = sum_iter + 1
            # sum_deviation = sum_deviation + deviation
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
            _, pred_e_index = tf.nn.top_k(pred_e, 1, name=None)
            f.write('pred_e: ' + '\t'.join([str(v) for v in tf.reshape(tf.squeeze(pred_e_index), [-1]).eval()]))
            f.write('\n')
            f.write('targ_e: ' + '\t'.join([str(v) for v in np.array(e_y_list[i]).flatten()]))
            f.write('\n')

<<<<<<< HEAD
=======
        # if i % (iterations // 10) == 0:
        # 	print('%f, precision: %f, deviation: %f' %(
        # 		i // (iterations // 10),
        # 		sum_correct_pred / (sum_iter * self.batch_size * self.length),
        # 		sum_deviation / (sum_iter * self.batch_size * self.length)))

>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
    def save_model(self, sess, logdir, counter):
        ckpt_file = '%s/model-%d.ckpt' % (logdir, counter)
        print('Checkpoint: %s' % ckpt_file)
        self.saver.save(sess, ckpt_file)


def get_config(config_mode):
    """Get model config."""

    if config_mode == "small":
        config = model_config.SmallConfig()
    elif config_mode == "medium":
        config = model_config.MediumConfig()
    elif config_mode == "large":
        config = model_config.LargeConfig()
    elif config_mode == "test":
        config = model_config.TestConfig()
    else:
        raise ValueError("Invalid model: %s", config_mode)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='small', type=str)
    parser.add_argument('--is_training', default=True, type=bool)
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--logdir', default='log/log_kick', type=str)
    parser.add_argument('--iters', default=30, type=int)
    parser.add_argument('--cell_type', default='T_GRUCell', type=str)
    args = parser.parse_args()

    assert args.logdir[-1] != '/'
<<<<<<< HEAD
    event_file = './T-pred-Dataset/lastfm-v5k_event2ID.txt'
    time_file = './T-pred-Dataset/lastfm-v5k_time2.txt'
    model_config = get_config(args.mode)
    is_training = args.is_training
    cell_type = args.cell_type
=======
    event_file = './T-pred-Dataset/RECSYS15_event.txt'
    time_file = './T-pred-Dataset/RECSYS15_time.txt'
    model_config = get_config(args.mode)
    is_training = args.is_training
    cell_type = args.cell_type
    print('vocab_size: ' + str(read_data.vocab_size(event_file)))
>>>>>>> 1eb9591939438650c7415df78dd8351748d03d00
    model = T_Pred(model_config, cell_type, event_file, time_file, is_training)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        if not args.eval_only:
            model.train(sess, args)
        model.eval(sess, args)


if __name__ == '__main__':
    main()
