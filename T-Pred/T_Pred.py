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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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
        self.length = config.output_length
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.lr = config.learning_rate
        self.LAMBDA = config.LAMBDA
        self.gamma = config.gamma
        self.event_to_id = read_data.build_vocab(self.event_file)
        self.train_data, self.valid_data, self.test_data = read_data.data_split(
            event_file, time_file, shuffle=True)
        self.embeddings = tf.get_variable(
            "embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
        self.sample_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps + self.length])
        self.target_t = tf.placeholder(tf.float32, [self.batch_size, self.length])
        self.inputs_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
        self.targets_e = tf.placeholder(tf.int64, [self.batch_size, self.length])
        self.input_e = tf.placeholder(tf.int64, [self.batch_size, self.num_steps])
        self.build()

    def encoder_e_t(self, cell_type, inputs, t):
        """
        Encode the inputs and timestamps into hidden representation.
        Using T_GRU cell
        """
        with tf.variable_scope("Generator/Event-Time"):
            outputs = utils.build_encoder_graph_t(
                cell_type,
                inputs,
                t,
                self.hidden_size,
                self.num_layers,
                self.batch_size,
                self.num_steps,
                self.keep_prob,
                self.is_training,
                "Encoder_et" + cell_type)
            hidden_r = tf.concat(outputs, 1)
            hidden_r = tf.reshape(hidden_r, [self.batch_size, self.num_steps, -1])
            return hidden_r

    def encoder_RecConv(self, cell_type, inputs, t):
        with tf.variable_scope('Generator/Event-Time'):
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

            inputs_t = tf.expand_dims(t, 2)
            output_t = utils.conv1d('G.T.Input', 1, self.filter_output_dim, self.filter_size, inputs_t)
            output_t = self.res_block('G.T.1', output_t)
            output_t = self.res_block('G.T.2', output_t)
            output_t = self.res_block('G.T.3', output_t)
            output_t = self.res_block('G.T.4', output_t)
            output_t = self.res_block('G.T.5', output_t)

            hidden_rt = tf.reshape(output_t, [-1, self.num_steps, self.filter_output_dim])
            # hidden_r = tf.concat([hidden_re, hidden_rt], 2)
            # hidden_r = tf.reshape(hidden_r, [self.batch_size, -1])
            return hidden_re, hidden_rt

    def modulator(self, hidden_re, hidden_rt, name):
        """
        attention to modulate the event and time
        hidden_re shape: [batch_size, num_steps, feature_size]
        hidden_rt shape: [batch_size, num_steps, feature_size]
        output: [batch_size, num_steps, feature_size]
        """
        z = tf.concat(
            [tf.multiply(tf.reshape(hidden_re, [self.batch_size, -1]), tf.reshape(hidden_rt, [self.batch_size, -1])),
             tf.reshape(hidden_re, [self.batch_size, -1]),
             tf.reshape(hidden_rt, [self.batch_size, -1])],
            1)
        with tf.variable_scope('Generator/M' + name):
            z_w = tf.get_variable("z_w", [z.get_shape()[1], 2], dtype=tf.float32)
            z_b = tf.get_variable("z_b", [2], dtype=tf.float32)
            logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
            b = tf.nn.softmax(logits_z)
            # b shape: [batch_size, 2]

            re = tf.expand_dims(tf.reshape(hidden_re, [self.batch_size, -1]), -1)
            rt = tf.expand_dims(tf.reshape(hidden_rt, [self.batch_size, -1]), -1)
            input_z = tf.transpose(tf.concat([re, rt], -1), [1, 0, 2])
            output_z = tf.multiply(b, input_z)
            output_z = tf.transpose(output_z, [1, 0, 2])
            output_z = tf.reduce_sum(output_z, 2)
            outputs_z = tf.reshape(output_z, [self.batch_size, self.num_steps, -1])

            # noise = make_noise(outputs_z.get_shape())
            # outputs_z = outputs_z + noise

            variable_b = tf.reduce_mean(b, 0)
            return outputs_z

    def g_event(self, hidden_r, name=''):
        """
        The generative model for time and event
        mode:
        1. use the concatenated hidden representation for each time step
        2. use the unfolded hidden representation separately for each time step
        """

        with tf.variable_scope("Generator/Event-g" + name):
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

    def g_time(self, hidden_r, name=''):
        """
        The generative model for time and event
        mode:
        1. use the concatenated hidden representation for each time step
        2. use the unfolded hidden representation separately for each time step
        """
        with tf.variable_scope('Generator/Time-g' + name):
            outputs = utils.build_rnn_graph_decoder1(
                hidden_r,
                self.num_layers,
                self.hidden_size,
                self.batch_size,
                self.length,
                "G_T.RNN")
            output = tf.reshape(tf.concat(outputs, 1), [-1, self.g_size])
            output = utils.linear('G_T.Output', self.g_size, 1, output)
            logits = tf.reshape(output, [self.batch_size, self.length, 1])
            return logits

    def attention_g_t(self, hidden_re, hidden_rt, num_gen):
        """
        If there are multiple generator for time sequences,
        use this attention vector to select an output from these generators.
        :param hidden_re: the representation of event
        :param hidden_rt: the representation of time
        :param num_gen: the number of generators for time sequence
        :return: the attention vector to weight the generators
        """
        with tf.variable_scope('Generator/Time/Attention'):
            hidden_re = tf.reshape(hidden_re, [self.batch_size, -1])
            hidden_rt = tf.reshape(hidden_rt, [self.batch_size, -1])
            a_w = tf.get_variable('a_w', [hidden_re.get_shape()[1] + hidden_rt.get_shape()[1], num_gen],
                                  dtype=tf.float32)
            a_b = tf.get_variable('a_b', [num_gen], dtype=tf.float32)
            logits_a = tf.nn.xw_plus_b(tf.concat([hidden_re, hidden_rt], 1), a_w, a_b)
            attention = tf.nn.softmax(logits_a, 1)
        return attention

    def discriminator(self, inputs_logits, num_blocks=3, use_bias=False, num_classes=1):
        """
        The discriminator to score the distribution of time and event
        If the time is consistent with the history times, give high score.
        If it is on the constant, give low score.
        Implementation:
        CNN"""
        with tf.variable_scope('Discriminator'):
            # inputs = tf.transpose(inputs_logits, [0,2,1])
            inputs = inputs_logits
            output = utils.conv1d('D.Input', 1, self.filter_output_dim, self.filter_size, inputs)
            output = self.res_block('D.1', output)
            output = self.res_block('D.2', output)
            output = self.res_block('D.3', output)
            output = self.res_block('D.4', output)
            output = self.res_block('D.5', output)
            output = tf.reshape(output, [-1, (self.length + self.num_steps) * self.filter_output_dim])
            # if the output size is 1, it is the discriminator score of D
            # if the output size is 2, it is a bi-classification result of D
            output = tf.nn.sigmoid(
                utils.linear('D.Output', (self.length + self.num_steps) * self.filter_output_dim, 1, output))
            print('The shape of output from D: ')
            print(output.get_shape())
            return output

    def res_block(self, name, inputs):
        output = inputs
        output = tf.nn.relu(output)
        output = utils.conv1d(name + '.1', self.filter_output_dim, self.filter_output_dim, self.filter_size, output)
        output = tf.nn.relu(output)
        output = utils.conv1d(name + '.2', self.filter_output_dim, self.filter_output_dim, self.filter_size, output)
        return inputs + (self.res_rate * output)

    def params_with_name(self, name):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [v for v in variables if name in v.name]

    def loss_with_wasserstein(self, pred_e, pred_t, real_e, real_t, input_t, sample_t):
        if self.cell_type == 'T_LSTMCell':
            variable_content_e = self.params_with_name('time_gate_t1')
        else:
            variable_content_e = None

        sample_t_concat = tf.expand_dims(sample_t, 2)
        pred_t_concat = tf.concat([tf.expand_dims(input_t, 2), pred_t], 1)
        disc_fake = self.discriminator(pred_t_concat)
        disc_real = self.discriminator(sample_t_concat)

        '''if the discriminator is a Wasserstein distance based critic'''
        disc_cost = -(tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake))
        # gen_t_cost_1 = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
        gen_t_cost_1 = -tf.reduce_mean(disc_fake)

        # WGANs lipschitz-penalty
        # delta = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)
        # difference = pred_t_concat - sample_t_concat
        # interploates = sample_t_concat + (delta * difference)
        # gradients = tf.gradients(self.discriminator(interploates), [interploates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        # gradient_penalty = tf.reduce_mean((slopes -1.)**2)
        # disc_cost += self.LAMBDA * gradient_penalty

        # Entropy for event sequence
        gen_e_cost = tf.contrib.seq2seq.sequence_loss(pred_e, real_e, weights=tf.ones([self.batch_size, self.length]),
                                                      name="SeqLoss")
        # huber loss for time sequence
        huber_t_loss = tf.losses.huber_loss(real_t, tf.exp(pred_t))

        gen_t_cost = gen_t_cost_1 + self.gamma * huber_t_loss
        gen_cost = gen_t_cost + self.alpha * gen_e_cost

        '''if the output of Discriminator is bi-classification, the losses used to train G and D is as follows'''
        # d_label_G = tf.one_hot(tf.ones([self.batch_size], dtype=tf.int32), 2)
        # d_label_real = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32), 2)
        #
        # disc_cost = tf.losses.log_loss(d_label_real, disc_real) + tf.losses.log_loss(d_label_G, disc_fake)
        # log_t_loss = tf.losses.huber_loss(real_t, tf.exp(pred_t))
        # gen_t_cost = tf.losses.log_loss(d_label_real, disc_fake) + log_t_loss
        #
        # gen_e_cost = tf.contrib.seq2seq.sequence_loss(
        # 	pred_e, real_e, weights=tf.ones([self.batch_size, self.length]), name="SeqLoss")
        #
        # gen_cost = gen_t_cost + self.alpha*gen_e_cost

        '''The separate training of Generator and Discriminator'''
        gen_params = self.params_with_name('Generator')
        gen_event_params = self.params_with_name('Event')
        disc_params = self.params_with_name('Discriminator')

        '''Use the Adam Optimizer to update the variables if we use gradient penalty'''
        # gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        # disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

        '''Use the RMSProp Optimizer to update the variables if we use basic wasserstein distance'''
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(disc_cost, var_list=disc_params)
        gen_event_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(gen_e_cost, var_list=gen_event_params)

        # constraint the weight of t for t1 to be negative
        if variable_content_e is not None:
            e_clip_op = []
            for v in variable_content_e:
                e_clip_op.append(tf.assign(v, tf.clip_by_value(v, -np.infty, 0)))
        else:
            e_clip_op = None

        # constraint the weight of discriminator between [-0.1, 0.1]
        # if we use gradient penalty for discriminator, there is no need to do weight clip!!!
        variable_content_w = self.params_with_name('Discriminator')
        if variable_content_w is not None:
            w_clip_op = []
            for v in variable_content_w:
                w_clip_op.append(tf.assign(v, tf.clip_by_value(v, -0.1, 0.1)))
        else:
            w_clip_op = None

        return gen_train_op, disc_train_op, w_clip_op, gen_cost, disc_cost, gen_t_cost, gen_e_cost, gen_t_cost_1, huber_t_loss, gen_event_op

    def build(self):
        """
        build the model
        define the loss function
        define the optimization method
        """
        self.targets_t = tf.expand_dims(self.target_t, 2)
        self.inputs_e = tf.nn.embedding_lookup(self.embeddings, self.input_e)

        hidden_t = self.encoder_e_t(self.cell_type, self.inputs_e, self.inputs_t)
        hidden_re, hidden_rt = self.encoder_RecConv(self.cell_type, self.inputs_e, self.inputs_t)
        output_re = self.modulator(hidden_re, hidden_rt, 'modulator/Event')
        output_rt = self.modulator(hidden_re, hidden_rt, 'modulator/Time')


        """
        The optional form of combination of output_re and output_rt
        1. modulator
        2. nothing
        3. concat
        4. plus
        """

        self.pred_e = self.g_event(tf.reshape(output_re, [self.batch_size, -1]))
        # self.pred_e = self.g_event(output_re)
        # use the extracted feature from input events and timestamps to generate the time sequence

        # take the prediction of events as input information for t_generators
        pred_e4t = utils.linear('Generator/pred_e4t.Iutput', self.vocab_size, self.g_size, self.pred_e)
        output_rt = tf.concat([tf.reshape(output_rt, [self.batch_size, -1]),
                               tf.reshape(pred_e4t, [self.batch_size, -1])], -1)

        if self.n_g == 1:
            self.pred_t = self.g_time(tf.reshape(output_rt, [self.batch_size, -1]))
        # self.pred_t = self.g_time(output_rt)
        # use random noise to generate the time sequence
        # self.pred_t = pred_t = self.g_time(make_noise(hidden_r.get_shape()))

        else:
            self.pred_t_list = []
            for i in range(self.n_g):
                pred_t = self.g_time(tf.reshape(output_rt, [self.batch_size, -1]), name=str(i))
                pred_t = tf.expand_dims(pred_t, 1)
                self.pred_t_list.append(pred_t)
            # self.pred_t = self.g_time(output_rt)
            # use random noise to generate the time sequence
            # self.pred_t = pred_t = self.g_time(make_noise(hidden_r.get_shape()))
            # the shape of pred_t_list: [batch_size, num_gen, predicted value]
            self.pred_t_list = tf.concat(self.pred_t_list, 1)

            '''The attention module for select the pred_t from all t-generators'''
            attention_gt = self.attention_g_t(output_re, output_rt, self.n_g)
            k = tf.argmax(attention_gt, 1)
            # print(k.get_shape())
            pred_t = []
            for i in range(self.batch_size):
                pred_t_e = self.pred_t_list[i][k[i]]
                pred_t.append(tf.expand_dims(pred_t_e, 0))
            self.pred_t = tf.concat(pred_t, 0)

        gen_train_op, disc_train_op, w_clip_op, gen_cost, disc_cost, gen_t_cost, gen_e_cost, gen_t_cost_1, huber_t_loss, g_e_op = self.loss_with_wasserstein(
            self.pred_e,
            self.pred_t,
            self.targets_e,
            self.targets_t,
            self.inputs_t,
            self.sample_t)

        self.g_train_op = gen_train_op
        self.d_train_op = disc_train_op
        self.w_clip_op = w_clip_op
        self.g_cost = gen_cost
        self.d_cost = disc_cost
        self.gen_e_cost = gen_e_cost
        self.gen_t_cost = gen_t_cost
        self.gen_t_cost_1 = gen_t_cost_1
        self.huber_t_loss = huber_t_loss
        self.g_e_op = g_e_op

        print('pred_e shape: ')
        print(self.pred_e.get_shape())
        print('targets_e shape: ')
        print(self.targets_e.get_shape())

        # MRR@k
        self.batch_precision, self.batch_precision_op = tf.metrics.average_precision_at_k(
            labels=self.targets_e, predictions=self.pred_e, k=20, name='precision_k')
        # Isolate the variables stored behind the scenes by the metric operation
        self.running_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_k")
        # Define initializer to initialize/reset running variables
        self.running_precision_vars_initializer = tf.variables_initializer(var_list=self.running_precision_vars)

        # Recall@k
        self.batch_recall, self.batch_recall_op = tf.metrics.recall_at_k(
            labels=self.targets_e, predictions=self.pred_e, k=20, name='recall_k')
        # Isolate the variables stored behind the scenes by the metric operation
        self.running_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_k")
        # Define initializer to initialize/reset running variables
        self.running_recall_vars_initializer = tf.variables_initializer(var_list=self.running_recall_vars)

        self.deviation = tf.reduce_sum(tf.abs(tf.squeeze(tf.exp(self.pred_t) - self.targets_t)))
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

            sess.run([self.running_precision_vars_initializer, self.running_recall_vars_initializer])
            # re initialize the metric variables of metric.precision and metric.recall,
            # to calculate these metric for each epoch
            batch_precision, batch_recall = 0.0, 0.0

            sum_iter = 0.0
            average_deviation, sum_deviation = 0.0, 0.0
            d_loss, g_loss, gen_e_cost, gen_t_cost, huber_t_loss = 0.0, 0.0, 0.0, 0.0, 0.0

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

            g_iters = 4
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

                if i % gap == 0:
                    _, _ = sess.run([self.d_train_op, self.w_clip_op], feed_dict=feed_dict)

                else:
                    _, _, _, deviation, batch_precision, batch_recall, d_loss, g_loss, gen_e_cost, gen_t_cost, huber_t_loss = sess.run(
                        [self.g_train_op, self.batch_precision_op, self.batch_recall_op,
                         self.deviation, self.batch_precision, self.batch_recall,
                         self.d_cost, self.g_cost, self.gen_e_cost, self.gen_t_cost, self.huber_t_loss], feed_dict=feed_dict)

                    sum_iter = sum_iter + 1
                    sum_deviation = sum_deviation + deviation
                    average_deviation = sum_deviation / (sum_iter * self.batch_size * self.length)

                # if self.cell_type == 'T_LSTMCell':
                #     sess.run(self.clip_op)

                if i % (batch_num // 10) == 0:
                    print('[epoch: %d, %d] precision: %f, recall: %f, deviation: %f' % (
                        epoch,
                        int(i // (batch_num // 10)),
                        batch_precision,
                        batch_recall,
                        average_deviation))
                    print('d_loss: %f, g_loss: %f, gen_e_loss: %f, gen_t_loss: %f, hunber_t_loss: %f' % (
                        d_loss, g_loss, gen_e_cost, gen_t_cost, huber_t_loss))

            '''evaluation'''
            sess.run([self.running_precision_vars_initializer, self.running_recall_vars_initializer])
            # re initialize the metric variables of metric.precision and metric.recall,
            # to calculate these metric for each epoch

            input_len, input_event_data, target_event_data, input_time_data, target_time_data = read_data.data_iterator(
                self.valid_data,
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

            sum_iter = 0.0
            sum_deviation = 0.0
            gen_cost_ratio = []
            t_cost_ratio = []

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

                _, _, deviation, batch_precision, batch_recall, d_loss, g_loss, gen_e_cost, gen_t_cost, gen_t_cost_1, huber_t_loss = sess.run(
                    [self.batch_precision_op, self.batch_recall_op, self.deviation, self.batch_precision, self.batch_recall,
                     self.d_cost, self.g_cost, self.gen_e_cost, self.gen_t_cost, self.gen_t_cost_1, self.huber_t_loss],
                    feed_dict=feed_dict)

                sum_iter = sum_iter + 1
                sum_deviation = sum_deviation + deviation
                gen_cost_ratio.append(gen_t_cost / gen_e_cost)
                t_cost_ratio.append(gen_t_cost_1 / huber_t_loss)

                if i % (batch_num // 10) == 0:
                    print('%d, precision: %f, recall: %f, deviation: %f, d_loss: %f, g_loss: %f, huber_t_loss:%f' % (
                        int(i // (batch_num // 10)),
                        batch_precision,
                        batch_recall,
                        sum_deviation / (sum_iter * self.batch_size * self.length),
                        d_loss,
                        g_loss,
                        huber_t_loss))
            self.alpha = tf.reduce_mean(gen_cost_ratio)
            self.gamma = tf.reduce_mean(t_cost_ratio)
            print('alpha: %f, gamma: %f' % (sess.run(self.alpha), sess.run(self.gamma)))

        self.save_model(sess, self.logdir, args.iters)

    def eval(self, sess, args):
        if not os.path.exists(args.logdir + '/output'):
            os.makedirs(args.logdir + '/output')

        if args.eval_only:
            self.test_data = read_data.load_test_dataset(self.dataset_file)

        if args.weights is not None:
            self.saver.restore(sess, args.weights)
            print_in_file("Saved")

        lr = self.learning_rate

        batch_size = 100

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

        f = open(os.path.join(args.logdir, "output.txt"), 'w+')

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
            pred_e, pred_t, = sess.run([self.pred_e, self.pred_t], feed_dict=feed_dict)

            # sum_correct_pred = sum_correct_pred + correct_pred
            # sum_iter = sum_iter + 1
            # sum_deviation = sum_deviation + deviation
            _, pred_e_index = tf.nn.top_k(pred_e, 1, name=None)
            f.write('pred_e: ' + '\t'.join([str(v) for v in tf.reshape(tf.squeeze(pred_e_index), [-1]).eval()]))
            f.write('\n')
            f.write('targ_e: ' + '\t'.join([str(v) for v in np.array(e_y_list[i]).flatten()]))
            f.write('\n')
            f.write('pred_t: ' + '\t'.join([str(v) for v in tf.exp(pred_t).eval()]))
            f.write('\n')
            f.write('targ_t: ' + '\t'.join([str(v) for v in np.array(t_y_list[i]).flatten()]))
            f.write('\n')

        # if i % (iterations // 10) == 0:
        # 	print('%f, precision: %f, deviation: %f' %(
        # 		i // (iterations // 10),
        # 		sum_correct_pred / (sum_iter * self.batch_size * self.length),
        # 		sum_deviation / (sum_iter * self.batch_size * self.length)))

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
    parser.add_argument('--iters', default=20, type=int)
    parser.add_argument('--cell_type', default='T_GRUCell', type=str)
    args = parser.parse_args()

    assert args.logdir[-1] != '/'
    event_file = './T-pred-Dataset/lastfm-v5k_event.txt'
    time_file = './T-pred-Dataset/lastfm-v5k_time.txt'
    model_config = get_config(args.mode)
    is_training = args.is_training
    cell_type = args.cell_type
    print('vocab_size: ' + str(read_data.vocab_size(event_file)))
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
