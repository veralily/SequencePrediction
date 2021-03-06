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
import logging
import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

event_file = './T-pred-Dataset/CIKM16_event.txt'
time_file = './T-pred-Dataset/CIKM16_time.txt'

FORMAT = "%(asctime)s - [line:%(lineno)s - %(funcName)10s() ] %(message)s"
DATA_TYPE = event_file.split('/')[-1].split('.')[0]
logging.basicConfig(filename='log/{}-{}-{}.log'.format('Pred-GAN-t', DATA_TYPE, str(datetime.datetime.now())),
            level=logging.INFO, format=FORMAT)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(FORMAT))
logging.getLogger().addHandler(handler)
logging.info('Start {}'.format(DATA_TYPE))

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
        self.length = 1 # config.output_length
        self.vocab_size = config.vocab_size
        self.learning_rate = config.learning_rate
        self.lr = config.learning_rate
        self.LAMBDA = config.LAMBDA
        self.gamma = config.gamma
        # self.event_to_id = read_data.build_vocab(self.event_file)
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
        with tf.variable_scope("Generator"):
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
            return hidden_r

    def encoder_t(self, t):
        with tf.variable_scope('Generator'):
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
            return hidden_rt

    def g_time(self, hidden_r, name=''):
        """
        The generative model for time and event
        mode:
        1. use the concatenated hidden representation for each time step
        2. use the unfolded hidden representation separately for each time step
        """
        with tf.variable_scope('Generator_T' + name):
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

    def attention_g_t(self, hidden_rt, num_gen):
        """
        If there are multiple generator for time sequences,
        use this attention vector to select an output from these generators.
        :param hidden_re: the representation of event
        :param hidden_rt: the representation of time
        :param num_gen: the number of generators for time sequence
        :return: the attention vector to weight the generators
        """
        with tf.variable_scope('Generator_T/Attention'):
            hidden_rt = tf.reshape(hidden_rt, [self.batch_size, -1])
            a_w = tf.get_variable('a_w', [hidden_rt.get_shape()[1], num_gen],
                                  dtype=tf.float32)
            a_b = tf.get_variable('a_b', [num_gen], dtype=tf.float32)
            logits_a = tf.nn.xw_plus_b(hidden_rt, a_w, a_b)
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
            logging.info('The shape of output from D: {}'.format(output.get_shape()))
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

    def loss_with_wasserstein(self, pred_t, real_t, input_t, sample_t):
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

        huber_t_loss = tf.losses.huber_loss(real_t, tf.exp(pred_t))
        gen_t_cost = gen_t_cost_1 + self.gamma * huber_t_loss

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
        disc_params = self.params_with_name('Discriminator')

        '''Use the Adam Optimizer to update the variables'''
        # gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        # disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

        '''Use the RMSProp Optimizer to update the variables'''
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(gen_t_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(disc_cost, var_list=disc_params)

        # constraint the weight of t for t1 to be negative
        if variable_content_e is not None:
            e_clip_op = []
            for v in variable_content_e:
                e_clip_op.append(tf.assign(v, tf.clip_by_value(v, -np.infty, 0)))
        else:
            e_clip_op = None

        # constraint the weight of discriminator between [-0.1, 0.1]
        variable_content_w = self.params_with_name('Discriminator')
        if variable_content_w is not None:
            w_clip_op = []
            for v in variable_content_w:
                w_clip_op.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        else:
            w_clip_op = None

        return gen_train_op, disc_train_op, w_clip_op, gen_t_cost, disc_cost, gen_t_cost_1, huber_t_loss

    def build(self):
        """
        build the model
        define the loss function
        define the optimization method
        """
        self.targets_t = tf.expand_dims(self.target_t, 2)
        self.inputs_e = tf.nn.embedding_lookup(self.embeddings, self.input_e)

        hidden_rt = self.encoder_t(self.inputs_t)

        if self.n_g == 1:
            self.pred_t = self.g_time(tf.reshape(hidden_rt, [self.batch_size, -1]))
        # self.pred_t = self.g_time(output_rt)
        # use random noise to generate the time sequence
        # self.pred_t = pred_t = self.g_time(make_noise(hidden_r.get_shape()))

        else:
            self.pred_t_list = []
            for i in range(self.n_g):
                pred_t = self.g_time(tf.reshape(hidden_rt, [self.batch_size, -1]), name=str(i))
                pred_t = tf.expand_dims(pred_t, 1)
                self.pred_t_list.append(pred_t)
            # self.pred_t = self.g_time(output_rt)
            # use random noise to generate the time sequence
            # self.pred_t = pred_t = self.g_time(make_noise(hidden_r.get_shape()))
            # the shape of pred_t_list: [batch_size, num_gen, predicted value]
            self.pred_t_list = tf.concat(self.pred_t_list, 1)

            '''The attention module for select the pred_t from all t-generators'''
            attention_gt = self.attention_g_t(hidden_rt, self.n_g)
            k = tf.argmax(attention_gt, 1)
            # print(k.get_shape())
            pred_t = []
            for i in range(self.batch_size):
                pred_t_e = self.pred_t_list[i][k[i]]
                pred_t.append(tf.expand_dims(pred_t_e, 0))
            self.pred_t = tf.concat(pred_t, 0)

        gen_train_op, disc_train_op, w_clip_op, gen_t_cost, disc_cost, gen_t_cost_1, huber_t_loss = self.loss_with_wasserstein(
            self.pred_t,
            self.targets_t,
            self.inputs_t,
            self.sample_t)

        self.g_train_op = gen_train_op
        self.d_train_op = disc_train_op
        self.w_clip_op = w_clip_op
        self.d_cost = disc_cost
        self.gen_t_cost = gen_t_cost
        self.gen_t_cost_1 = gen_t_cost_1
        self.huber_t_loss = huber_t_loss

        self.deviation = tf.reduce_mean(tf.abs(tf.squeeze(tf.exp(self.pred_t) - self.targets_t)))
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

            sum_iter = 0.0
            average_deviation, sum_deviation = 0.0, 0.0
            d_loss, gen_t_cost, huber_t_loss = 0.0, 0.0, 0.0

            i_e, t_e, i_t, t_t = read_data.data_iterator(self.train_data, self.num_steps, self.length)

            sample_t = read_data.generate_sample_t(self.batch_size, i_t, t_t)
            batch_num = len(list(read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t)))
            logging.info('Training batch num {}'.format(batch_num))

            g_iters = 5
            gap = g_iters + 1
            i = 0

            for e_x, e_y, t_x, t_y in read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t):

                feed_dict = {
                    self.input_e: e_x,
                    self.inputs_t: np.maximum(np.log(t_x), 0),
                    self.target_t: t_y,
                    self.targets_e: e_y,
                    self.sample_t: np.maximum(np.log(sample_t), 0)}

                if i > 0 and i % (batch_num // 10) == 0:
                    self.lr = self.lr * 2. / 3

                if i % gap == 0:
                    _, _ = sess.run([self.d_train_op, self.w_clip_op], feed_dict=feed_dict)

                else:
                    _, deviation, d_loss, gen_t_cost, huber_t_loss = sess.run(
                        [self.g_train_op, self.deviation,
                         self.d_cost, self.gen_t_cost, self.huber_t_loss], feed_dict=feed_dict)

                    sum_iter = sum_iter + 1
                    sum_deviation = sum_deviation + deviation
                    average_deviation = sum_deviation / sum_iter

                # if self.cell_type == 'T_LSTMCell':
                #     sess.run(self.clip_op)

                if i % (batch_num // 10) == 0:
                    logging.info('[epoch: {}, {}] deviation: {}'.format (
                        epoch,
                        int(i // (batch_num // 10)),
                        average_deviation))
                    logging.info('d_loss: {}, gen_t_loss: {}, hunber_t_loss: {}'.format (
                        d_loss, gen_t_cost, huber_t_loss))
                i += 1

            '''evaluation'''

            i_e, t_e, i_t, t_t = read_data.data_iterator(
                self.valid_data,
                self.num_steps,
                self.length)

            sample_t = read_data.generate_sample_t(
                self.batch_size,
                i_t,
                t_t)

            batch_num = len(list(read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t)))
            logging.info('Evaluation Batch Num {}'.format(batch_num))

            sum_iter = 0.0
            sum_deviation = 0.0
            gen_cost_ratio = []
            t_cost_ratio = []
            i = 0

            self.lr = self.learning_rate

            for e_x, e_y, t_x, t_y in read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t):
                feed_dict = {
                    self.input_e: e_x,
                    self.inputs_t: np.maximum(np.log(t_x), 0),
                    self.target_t: t_y,
                    self.targets_e: e_y,
                    self.sample_t: np.maximum(np.log(sample_t), 0)}

                if i > 0 and i % (batch_num // 10) == 0:
                    self.lr = self.lr * 2. / 3

                deviation, d_loss, gen_t_cost, gen_t_cost_1, huber_t_loss = sess.run(
                    [self.deviation, self.d_cost, self.gen_t_cost, self.gen_t_cost_1, self.huber_t_loss],
                    feed_dict=feed_dict)

                sum_iter = sum_iter + 1
                sum_deviation = sum_deviation + deviation
                t_cost_ratio.append(gen_t_cost_1 / huber_t_loss)

                if i % (batch_num // 10) == 0:
                    logging.info('{} deviation: {}, d_loss: {}, g_loss: {}, huber_t_loss:{}'.format (
                        int(i // (batch_num // 10)),
                        sum_deviation / sum_iter,
                        d_loss,
                        gen_t_cost,
                        huber_t_loss))
                i += 1
            self.gamma = tf.reduce_mean(t_cost_ratio)
            logging.info('gamma: {}'.format (sess.run(self.gamma)))

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

        i_e, t_e, i_t, t_t = read_data.data_iterator(
            self.test_data,
            self.num_steps,
            self.length)

        sample_t = read_data.generate_sample_t(
            self.batch_size,
            i_t,
            t_t)

        batch_num = len(list(read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t)))
        logging.info('Evaluation Batch Num {}'.format(batch_num))

        f = open(os.path.join(args.logdir, "output_t.txt"), 'w+')
        i = 0

        for e_x, e_y, t_x, t_y in read_data.generate_batch(self.batch_size, i_e, t_e, i_t, t_t):
            feed_dict = {
                # self.input_e: e_x,
                self.inputs_t: np.maximum(np.log(t_x), 0),
                self.target_t: t_y,
                # self.targets_e: e_y,
                self.sample_t: np.maximum(np.log(sample_t), 0)}

            if i > 0 and i % (batch_num // 10) == 0:
                lr = lr * 2. / 3
            # correct_pred, deviation, pred_e, pred_t, d_loss, g_loss, gen_e_cost, gen_t_cost = sess.run(
            # 	[self.correct_pred, self.deviation, self.pred_e, self.pred_t, self.d_cost, self.g_cost,
            # 	self.gen_e_cost, self.gen_t_cost,self.disc_cost_1, self.gradient_penalty],
            # 	feed_dict = feed_dict)
            pred_t = sess.run(self.pred_t, feed_dict=feed_dict)

            # sum_correct_pred = sum_correct_pred + correct_pred
            # sum_iter = sum_iter + 1
            # sum_deviation = sum_deviation + deviation
            f.write('pred_t: ' + '\t'.join([str(v) for v in tf.exp(pred_t).eval()]))
            f.write('\n')
            f.write('targ_t: ' + '\t'.join([str(v) for v in np.array(t_y).flatten()]))
            f.write('\n')

            i += 1

        # if i % (iterations // 10) == 0:
        # 	print('%f, precision: %f, deviation: %f' %(
        # 		i // (iterations // 10),
        # 		sum_correct_pred / (sum_iter * self.batch_size * self.length),
        # 		sum_deviation / (sum_iter * self.batch_size * self.length)))

    def save_model(self, sess, logdir, counter):
        ckpt_file = '%s/model-%d.ckpt' % (logdir, counter)
        logging.info('Checkpoint: {}'.format(ckpt_file))
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
    model_config = get_config(args.mode)
    is_training = args.is_training
    cell_type = args.cell_type
    # print('vocab_size: ' + str(read_data.vocab_size(event_file)))
    model = T_Pred(model_config, cell_type, event_file, time_file, is_training)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        if not args.eval_only:
            model.train(sess, args)
        model.eval(sess, args)
    logging.info('Logging End!')


if __name__ == '__main__':
    main()
