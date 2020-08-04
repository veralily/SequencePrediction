import tensorflow as tf
from modules import *

class MM_CPred():
    def __init__(self, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.vocab_size = args.vocab_size
        self.alpha = tf.placeholder(tf.float32)
        self.gamma = tf.placeholder(tf.float32)
        self.num_units = args.hidden_size
        self.len = args.len
        self.num_gen_t = args.num_gen_t
        self.sample_t = tf.placeholder(tf.float32, shape=(None, args.T+args.len))
        self.target_t = tf.placeholder(tf.float32, shape=(None, args.len))
        self.inputs_t = tf.placeholder(tf.float32, shape=(None, args.T))
        self.target_e = tf.placeholder(tf.int64, shape=(None, args.len))
        self.inputs_e = tf.placeholder(tf.int64, shape=(None, args.T))
        self.logits_e, self.logits_t, self.l_logits_t = self.build()
        self.train_event_op = self.train_event(args.lr_e)
        self.train_time_op = self.train_time(args.lr_t)
        self.train_gen_op, self.train_disc_op, self.train_w_clip_op = self.joint_train(args.lr_j)
        self.cross_entropy_loss = self.event_loss(self.logits_e, self.target_e)
        self.huber_loss = self.time_loss(self.logits_t, self.target_t)
        self.gen_t_loss, self.disc_t_loss = self.loss_with_wasserstein(self.inputs_t, self.logits_t, self.sample_t)

    def Enc_e(self, inputs, num_units, scope='Pred/Event/Enc'):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            # Self-attention layer
            outputs = multihead_attention(queries=outputs,
                                          keys=outputs,
                                          num_units = num_units)
        return outputs


    def Enc_t(self, inputs, num_units, scope="Pred/Time/Enc"):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.variable_scope(scope):
            outputs = inputs
            outputs = conv1d(outputs, scope= 'G.T.Conv1D', reuse=True, num_units=num_units)
            outputs = res_block('G.T.1', outputs)
            outputs = res_block('G.T.2', outputs)
            outputs = res_block('G.T.3', outputs)
            outputs = res_block('G.T.4', outputs)
            outputs = res_block('G.T.5', outputs)
            # Self-attention layer
            outputs = multihead_attention(queries=outputs,
                                          keys=outputs,
                                          num_units=num_units)
        return outputs


    def Gen_e(self, inputs, num_units, vocab_size, scope='Pred/Event/Gen'):
        """
        :param inputs: A Tensor. Hidden representation for event generator. (N, olen, C)
        :param num_units: An int.
        :param vocab_size: An int. The number of vocabulary.
        :param scope: A Str.
        :return: outputs: A Tensor. (N, olen, C). logits: A Tensor. (N, olen, vocab_size).
        """
        with tf.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            logits = tf.keras.layers.Dense(units=vocab_size, activation='softmax')(outputs)

        return outputs, logits


    def Gen_t(self, inputs, num_units, scope='Pred/Time/Gen'):
        """
        :param inputs: A Tensor. Hidden representation for time generator. (N, olen, C)
        :param num_units: An int.
        :param scope: A str.
        :return:  A Tensor. (N, olen, C). logits: A Tensor. (N, olen, 1).
        """
        with tf.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            logits = tf.keras.layers.Dense(1, activation='relu')(outputs)
        return outputs, logits


    def M_Gen_t(self, inputs, num_units, num_gen_t, scope='Multiple'):
        with tf.variable_scope(scope):
            outputs_list = []
            logits_list = []
            for i in num_gen_t:
                outputs, logits = self.Gen_t(inputs, num_units=num_units, scope='Pred/Time/Gen_'+str(i+1))
                outputs_list.append(outputs)
                logits_list.append(logits)
        return tf.convert_to_tensor(tf.concat(outputs_list,axis=2)), tf.convert_to_tensor(tf.concat(logits_list,axis=2))


    def selector(self, inputs, num_gen_t, scope='Pred/Time/Sel'):
        """
        :param inputs: A tensor. Stacked representation for event and time. (N, T, 2C)
        :return: Gumbel-softmaxed attention for multiple time generators.
        """
        with tf.variable_scope(scope):
            logits = tf.keras.layers.Dense(num_gen_t)(inputs)
            # Gumbel softmax
            attention, index = gumbel_softmax(logits, axis=-1)
        return attention, index


    def output_t(self, inputs, sel_index, scope='Pred/Time/Out'):
        """
        :param inputs: A Tensor. (N, T, n)
        :param sel_index: A Tensor. (N, T) index of selector
        :param scope: A Str
        :return: Weighted sum form of one-hot index and inputs.
        """
        with tf.variable_scope(scope):
            weights = tf.one_hot(sel_index, depth=tf.shape(inputs)[0])
            outputs = tf.reduce_sum(tf.multiply(inputs,weights),axis=-1)
        return  outputs

    def discriminator(self, inputs, num_units, res_rate = 0.2, scope='Disc/t'):
        """
        The discriminator to score the distribution of time and event
        If the time is consistent with the history times, give high score.
        If it is on the constant, give low score.
        Implementation:
        CNN
        :param inputs: A Tensor. (N, L, 1)
        :param num_units: An int
        :return A Tensor. (N,1)"""
        with tf.variable_scope(scope):
            outputs = inputs
            outputs = conv1d(outputs, num_units,scope='Disc/Conv1d')
            outputs = res_block(outputs,num_units, res_rate, scope='Disc/ResBlock_1')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_2')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_3')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_4')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_5')

            # if the output size is 1, it is the discriminator score of D
            # if the output size is 2, it is a bi-classification result of D
            logits = tf.keras.layers.Dense(1,activation='sigmoid')(outputs)
            logits = tf.keras.layers.Dense(1)(tf.squeeze(logits))
            # logging.info('The shape of output from D {}'.format(output.get_shape()))
            return logits


    def params_with_name(self, name):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [v for v in variables if name in v.name]


    def event_loss(self, outputs_e, target_e):
        """
        :param outputs_e: A Tensor. (N, T, vocab_size)
        :param target_e: A tensor. (N, T, vocab_size)
        :return:
        """
        # Entropy for event sequence
        cross_entropy_loss = tf.losses.softmax_cross_entropy(logits=outputs_e,
                                                             onehot_labels=target_e,
                                                             scope='SeqLoss_e',
                                                             reduction=None)
        return tf.reduce_mean(cross_entropy_loss)


    def time_loss(self, outputs_t, target_t):
        """
        :param outputs_t:
        :param target_t:
        :return:
        """
        # Huber loss for time sequence
        huber_loss = tf.losses.huber_loss(labels=target_t,
                                          predictions=outputs_t,
                                          scope='HuberLoss_t',
                                          reduction=None)
        huber_loss = tf.reduce_mean(huber_loss)
        return huber_loss


    def loss_with_wasserstein(self, inputs_t, outputs_t, sample_t):
        """
        :param inputs_t: A Tensor. (N, T, 1)
        :param outputs_t: A Tensor. (N, len, 1)
        :param sample_t: A Tensor. (N, T+len, 1)
        :return:
        """

        disc_fake = self.discriminator(tf.concat([inputs_t, outputs_t], axis=1))
        disc_real = self.discriminator(sample_t)

        '''if the discriminator is a Wasserstein distance based critic'''
        disc_cost = -(tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake))
        # gen_t_cost_1 = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
        gen_cost = -tf.reduce_mean(disc_fake)

        # WGANs lipschitz-penalty
        # delta = tf.random_uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=1.)
        # difference = pred_t_concat - sample_t_concat
        # interploates = sample_t_concat + (delta * difference)
        # gradients = tf.gradients(self.discriminator(interploates), [interploates])[0]
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
        # gradient_penalty = tf.reduce_mean((slopes -1.)**2)
        # disc_cost += self.LAMBDA * gradient_penalty

        '''if the output of Discriminator is bi-classification, the losses used to train G and D is as follows'''
        # d_label_G = tf.one_hot(tf.ones([self.batch_size], dtype=tf.int32), 2)
        # d_label_real = tf.one_hot(tf.zeros([self.batch_size], dtype=tf.int32), 2)
        #
        # disc_cost = tf.losses.log_loss(d_label_real, disc_real) + tf.losses.log_loss(d_label_G, disc_fake)
        # gen_t_cost = tf.losses.log_loss(d_label_real, disc_fake)
        return disc_cost, gen_cost


    def joint_loss(self, inputs_t, outputs_t, target_t, sample_t, outputs_e, target_e):
        disc_cost, gen_cost = self.loss_with_wasserstein(inputs_t, outputs_t, sample_t)
        cross_entropy_loss = self.event_loss(outputs_e, target_e)
        huber_loss = self.time_loss(outputs_e, target_t)
        # Train generator
        gen_cost = gen_cost + self.gamma * huber_loss + self.alpha * cross_entropy_loss

        # Train discriminator
        disc_cost = disc_cost

        return gen_cost, disc_cost


    def build(self):
        """
        :return:
        """
        # Embedding
        inputs_e = embedding(self.inputs_e, self.vocab_size, self.num_units)
        # Encode events (N, T, C)
        outputs_e = self.Enc_e(inputs_e, self.num_units)
        # Encode times (N, T, C)
        outputs_t = self.Enc_t(self.inputs_t, self.num_units)
        # Decode events
        hidden_e = tf.tile(tf.reduce_sum(outputs_e, axis=1),[1, self.len, 1]) # (N, len, C)
        outputs_e, logits_e = self.Gen_e(hidden_e, self.num_units, self.vocab_size)
        # Decode times
        hidden_t = tf.tile(tf.reduce_sum(outputs_t, axis=1), [1, self.len, 1])
        hidden_t = tf.concat([tf.tile(tf.reduce_sum(outputs_e, axis=1), [1, self.len, 1]), hidden_t], axis=-1)
        if self.num_gen_t != 1:
            outputs_t, logits_t = self.M_Gen_t(hidden_t, self.num_units, self.num_gen_t)
            sel_weights, sel_index = self.selector(hidden_t,self.num_gen_t)
            logits_t = self.output_t(logits_t, sel_index)
            l_logits_t = self.output_t(logits_t, sel_weights)
        else:
            outputs_t, logits_t = self.Gen_t(hidden_t, self.num_units)
            l_logits_t = logits_t
        return logits_e, logits_t, l_logits_t


    def train_event(self, lr):
        gen_e_params = self.params_with_name('Event')
        event_cross_entropy = self.event_loss(self.logits_e, self.target_e)
        train_event_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(event_cross_entropy,
                                                                              var_list=gen_e_params)
        return train_event_op

    def train_time(self, lr):
        gen_t_params = self.params_with_name('Time')
        time_huber_loss = self.time_loss(self.l_logits_t, self.target_t)
        train_time_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(time_huber_loss,
                                                                             var_list=gen_t_params)
        return train_time_op


    def joint_train(self, lr):
        gen_params = self.params_with_name('Pred')
        disc_params = self.params_with_name('Disc')
        gen_loss, disc_loss = self.joint_loss(inputs_t=self.inputs_t,
                                              outputs_t=self.l_logits_t,
                                              target_t = self.target_t,
                                              sample_t = self.sample_t,
                                              outputs_e = self.logits_e,
                                              target_e = self.target_e)
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(gen_loss,var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(disc_loss,var_list=disc_params)

        # constraint the weight of discriminator between [-0.1, 0.1]
        # if we use gradient penalty for discriminator, there is no need to do weight clip!!!
        variable_content_w = self.params_with_name('Disc')
        if variable_content_w is not None:
            w_clip_op = []
            for v in variable_content_w:
                w_clip_op.append(tf.assign(v, tf.clip_by_value(v, -0.1, 0.1)))
        else:
            w_clip_op = None
        return gen_train_op, disc_train_op, w_clip_op