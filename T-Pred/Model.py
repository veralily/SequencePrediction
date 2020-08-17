import tensorflow as tf
from modules import *

class MM_CPred():
    def __init__(self, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.vocab_size = args.vocab_size
        self.alpha = tf.compat.v1.placeholder(tf.float32)
        self.gamma = tf.compat.v1.placeholder(tf.float32)
        self.num_units = args.hidden_size
        self.len = args.len
        self.num_gen_t = args.num_gen_t
        self.res_rate = args.res_rate
        self.sample_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.T+args.len))
        self.target_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.len))
        self.inputs_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.T))
        self.target_e = tf.compat.v1.placeholder(tf.int64, shape=(None, args.len))
        self.inputs_e = tf.compat.v1.placeholder(tf.int64, shape=(None, args.T))
        self.logits_e, self.logits_t, self.l_logits_t, self.gen_t_loss, self.disc_t_loss = self.build()
        self.cross_entropy_loss = self.event_loss(self.logits_e, self.target_e)
        self.huber_loss = self.time_loss(self.logits_t, self.target_t)
        self.train_event_op = self.train_event(args.lr_e)
        self.train_time_op = self.train_time(args.lr_t)
        self.train_gen_op, self.train_disc_op, self.train_w_clip_op = self.joint_train(args.lr_j)
        self.MAE, self.MAE_op, self.running_MAE_vars_initializer, \
        self.precision, self.precision_op, self.running_precision_vars_initializer, \
        self.recall, self.recall_op, self.running_recall_vars_initializer = self.cal_metrics(metric_k=args.metric_k)


    def Enc_e(self, inputs, num_units, scope='Pred/Event/Enc'):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.compat.v1.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            # Self-attention layer
            outputs = multihead_attention(queries=outputs,
                                          keys=outputs,
                                          num_units=num_units)
        return outputs


    def Enc_t(self, inputs, num_units, scope="Pred/Time/Enc"):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.compat.v1.variable_scope(scope):
            outputs = inputs
            outputs = conv1d(outputs, scope='G.T.Conv1D', num_units=num_units)
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.1')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.2')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.3')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.4')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.5')
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
        with tf.compat.v1.variable_scope(scope):
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
        with tf.compat.v1.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            logits = tf.keras.layers.Dense(1, activation='relu')(outputs)
        return outputs, logits


    def M_Gen_t(self, inputs, num_units, num_gen_t, scope='Multiple'):
        with tf.compat.v1.variable_scope(scope):
            outputs_list = []
            logits_list = []
            for i in range(num_gen_t):
                outputs, logits = self.Gen_t(inputs, num_units=num_units, scope='Pred/Time/Gen_'+str(i+1))
                outputs_list.append(outputs)
                logits_list.append(logits)
        return tf.convert_to_tensor(tf.concat(outputs_list, axis=2)),\
               tf.convert_to_tensor(tf.concat(logits_list, axis=2))


    def selector(self, inputs, num_gen_t, scope='Pred/Time/Sel'):
        """
        :param inputs: A tensor. Stacked representation for event and time. (N, T, 2C)
        :return: Gumbel-softmaxed attention for multiple time generators.
        """
        with tf.compat.v1.variable_scope(scope):
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
        with tf.compat.v1.variable_scope(scope):
            weights = tf.one_hot(sel_index, depth=tf.shape(inputs)[-1], dtype=tf.float32)
            outputs = tf.reduce_sum(tf.multiply(inputs, weights), axis=-1)
        return outputs

    def discriminator(self, inputs, num_units, res_rate = 0.2, scope='Disc/t'):
        """
        The discriminator to score the distribution of time and event
        If the time is consistent with the history times, give high score.
        If it is on the constant, give low score.
        Implementation:
        CNN
        :param inputs: A Tensor. (N, L)
        :param num_units: An int
        :return A Tensor. (N, 1)"""
        with tf.compat.v1.variable_scope(scope, reuse=True):
            outputs = tf.expand_dims(inputs, axis=2)
            outputs = conv1d(outputs, num_units, scope='Disc/Conv1d')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_1')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_2')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_3')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_4')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_5')

            # if the output size is 1, it is the discriminator score of D
            # if the output size is 2, it is a bi-classification result of D
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
            outputs = tf.squeeze(outputs, axis=-1)
            logits = tf.keras.layers.Dense(1)(outputs)
            # logging.info('The shape of output from D {}'.format(output.get_shape()))
            return logits


    def params_with_name(self, name):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [v for v in variables if name in v.name]


    def event_loss(self, outputs_e, target_e):
        """
        :param outputs_e: A Tensor. (N, T, vocab_size)
        :param target_e: A tensor. (N, T)
        :return:
        """
        # Entropy for event sequence
        target_e = tf.one_hot(target_e, depth=outputs_e.get_shape()[-1])
        cross_entropy_loss = tf.losses.softmax_cross_entropy(logits=outputs_e,
                                                             onehot_labels=target_e,
                                                             scope='SeqLoss_e')
        return tf.reduce_mean(cross_entropy_loss)


    def time_loss(self, outputs_t, target_t):
        """
        :param outputs_t: A Tensor. (N, len)
        :param target_t: A Tensor. ()
        :return:
        """
        # Huber loss for time sequence
        huber_loss = tf.compat.v1.losses.huber_loss(labels=target_t,
                                                    predictions=outputs_t,
                                                    scope='HuberLoss_t')
        huber_loss = tf.reduce_mean(huber_loss)
        return huber_loss


    def loss_with_wasserstein(self, inputs_t, outputs_t, sample_t):
        """
        :param inputs_t: A Tensor. (N, T)
        :param outputs_t: A Tensor. (N, len)
        :param sample_t: A Tensor. (N, T+len)
        :return:
        """
        pred_t = tf.concat([inputs_t, outputs_t], axis=1)
        disc_fake = self.discriminator(pred_t, num_units=self.num_units)
        disc_real = self.discriminator(sample_t, self.num_units)

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


    def joint_loss(self):
        cross_entropy_loss = self.cross_entropy_loss
        huber_loss = self.huber_loss
        # Train generator
        gen_cost = self.gen_t_loss + self.gamma * huber_loss + self.alpha * cross_entropy_loss

        # Train discriminator
        disc_cost = self.disc_t_loss

        return gen_cost, disc_cost


    def build(self):
        """
        :return:
        """
        # Embedding
        inputs_e, embedding_table  = embedding(self.inputs_e, self.vocab_size, self.num_units)
        # Encode events (N, T, C)
        outputs_e = self.Enc_e(inputs_e, self.num_units)  # (num_heads, N, T, C)
        # Encode times (N, T, C)
        inputs_t = tf.expand_dims(self.inputs_t, axis=2)
        outputs_t = self.Enc_t(inputs=inputs_t, num_units=self.num_units)
        # Decode events
        hidden_e = tf.tile(tf.expand_dims(tf.reduce_sum(outputs_e, axis=1), axis=1), [1, self.len, 1])  # (N, len, C)
        outputs_e, logits_e = self.Gen_e(hidden_e, self.num_units, self.vocab_size)
        # Decode times
        hidden_t = tf.tile(tf.expand_dims(tf.reduce_sum(outputs_t, axis=1),axis=1), [1, self.len, 1])
        hidden_t = tf.concat([tf.tile(tf.expand_dims(tf.reduce_sum(outputs_e, axis=1), axis=1),
                                      [1, self.len, 1]), hidden_t], axis=-1)
        if self.num_gen_t != 1:
            outputs_t, logits_t = self.M_Gen_t(hidden_t, self.num_units, self.num_gen_t)
            sel_weights, sel_index = self.selector(hidden_t, self.num_gen_t)
            l_logits_t = tf.reduce_sum(tf.multiply(logits_t, sel_weights), axis=-1)
            logits_t = self.output_t(logits_t, sel_index)
        else:
            outputs_t, logits_t = self.Gen_t(hidden_t, self.num_units)
            l_logits_t = logits_t
        disc_t_cost, gen_t_cost = self.loss_with_wasserstein(self.inputs_t, l_logits_t, self.sample_t)
        return logits_e, logits_t, l_logits_t, disc_t_cost, gen_t_cost


    def train_event(self, lr):
        gen_e_params = self.params_with_name('Event')
        # print('---gen e params')
        # print(gen_e_params)
        event_cross_entropy = self.event_loss(self.logits_e, self.target_e)
        train_event_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(event_cross_entropy,
                                                                              var_list=gen_e_params)
        return train_event_op

    def train_time(self, lr):
        gen_t_params = self.params_with_name('Time')
        # print('---gen t params')
        # print(gen_t_params)
        time_huber_loss = self.time_loss(self.l_logits_t, self.target_t)
        train_time_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(time_huber_loss,
                                                                             var_list=gen_t_params)
        return train_time_op


    def joint_train(self, lr):
        gen_params = self.params_with_name('Pred')
        # print(gen_params)
        disc_params = self.params_with_name('Disc')
        # print(disc_params)
        gen_loss, disc_loss = self.joint_loss()
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(gen_loss, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(disc_loss, var_list=disc_params)

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


    def cal_metrics(self, metric_k):
        # Define Metrics
        metric_k= metric_k
        # MRR@k
        precision, precision_op = tf.metrics.precision_at_k(labels=self.target_e,
                                                            predictions=self.logits_e,
                                                            k=metric_k, name='precision_k')
        # Isolate the variables stored behind the scenes by the metric operation
        running_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_k")
        # Define initializer to initialize/reset running variables
        running_precision_vars_initializer = tf.variables_initializer(var_list=running_precision_vars)

        # Recall@k
        recall, recall_op = tf.metrics.recall_at_k(labels=self.target_e,
                                                   predictions=self.logits_e,
                                                   k=metric_k,
                                                   name='recall_k')
        # Isolate the variables stored behind the scenes by the metric operation
        running_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_k")
        # Define initializer to initialize/reset running variables
        running_recall_vars_initializer = tf.variables_initializer(var_list=running_recall_vars)

        # MAE
        MAE, MAE_op = tf.metrics.mean_absolute_error(labels=self.target_t,
                                                     predictions=self.logits_t,
                                                     name='MAE')
        # Isolate the variables stored behind the scenes by the metric operation
        running_MAE_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="MAE")
        # Define initializer to initialize/reset running variables
        running_MAE_vars_initializer = tf.variables_initializer(var_list=running_MAE_vars)

        return MAE, MAE_op, running_MAE_vars_initializer,\
               precision, precision_op, running_precision_vars_initializer,\
               recall, recall_op, running_recall_vars_initializer



class MM_CPred_2():
    """
    concat the R_e and R_t as the hidden representation for generators ranther ourpur_e and R_t
    replace the 3-step training with 2 step training
    """
    def __init__(self, args, reuse=None):
        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
        self.vocab_size = args.vocab_size
        self.alpha = tf.compat.v1.placeholder(tf.float32)
        self.gamma = tf.compat.v1.placeholder(tf.float32)
        self.num_units = args.hidden_size
        self.len = args.len
        self.num_gen_t = args.num_gen_t
        self.res_rate = args.res_rate
        self.sample_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.T+args.len))
        self.target_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.len))
        self.inputs_t = tf.compat.v1.placeholder(tf.float32, shape=(None, args.T))
        self.target_e = tf.compat.v1.placeholder(tf.int64, shape=(None, args.len))
        self.inputs_e = tf.compat.v1.placeholder(tf.int64, shape=(None, args.T))
        self.logits_e, self.logits_t, self.l_logits_t, self.gen_t_loss, self.disc_t_loss = self.build()
        self.cross_entropy_loss = self.event_loss(self.logits_e, self.target_e)
        self.huber_loss = self.time_loss(self.logits_t, self.target_t)
        self.train_MLE_op = self.train_MLE(args.lr_e)
        self.train_gen_op, self.train_disc_op, self.train_w_clip_op = self.train_GAN(args.lr_j)
        self.MAE, self.MAE_op, self.running_MAE_vars_initializer, \
        self.precision, self.precision_op, self.running_precision_vars_initializer, \
        self.recall, self.recall_op, self.running_recall_vars_initializer = self.cal_metrics(metric_k=args.metric_k)


    def Enc_e(self, inputs, num_units, scope='Pred/Event/Enc'):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.compat.v1.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            # Self-attention layer
            outputs = multihead_attention(queries=outputs,
                                          keys=outputs,
                                          num_units=num_units)
        return outputs


    def Enc_t(self, inputs, num_units, scope="Pred/Time/Enc"):
        """

        :param inputs: A Tensor. (N, T, C)
        :param num_units: An int. The number of dimensions.
        :param scope: A str.
        :return: A Tensor. (N, T, num_units)
        """
        with tf.compat.v1.variable_scope(scope):
            outputs = inputs
            outputs = conv1d(outputs, scope='G.T.Conv1D', num_units=num_units)
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.1')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.2')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.3')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.4')
            outputs = res_block(inputs=outputs,num_units=self.num_units,res_rate=self.res_rate,scope='G.T.5')
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
        with tf.compat.v1.variable_scope(scope):
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
        with tf.compat.v1.variable_scope(scope):
            gru = tf.keras.layers.GRU(num_units,
                                      return_sequences=True,
                                      return_state=True)
            outputs, final_states = gru(inputs)
            logits = tf.keras.layers.Dense(1, activation='relu')(outputs)
        return outputs, logits


    def M_Gen_t(self, inputs, num_units, num_gen_t, scope='Multiple'):
        with tf.compat.v1.variable_scope(scope):
            outputs_list = []
            logits_list = []
            for i in range(num_gen_t):
                outputs, logits = self.Gen_t(inputs, num_units=num_units, scope='Pred/Time/Gen_'+str(i+1))
                outputs_list.append(outputs)
                logits_list.append(logits)
        return tf.convert_to_tensor(tf.concat(outputs_list, axis=2)),\
               tf.convert_to_tensor(tf.concat(logits_list, axis=2))


    def selector(self, inputs, num_gen_t, scope='Pred/Time/Sel'):
        """
        :param inputs: A tensor. Stacked representation for event and time. (N, T, 2C)
        :return: Gumbel-softmaxed attention for multiple time generators.
        """
        with tf.compat.v1.variable_scope(scope):
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
        with tf.compat.v1.variable_scope(scope):
            weights = tf.one_hot(sel_index, depth=tf.shape(inputs)[-1], dtype=tf.float32)
            outputs = tf.reduce_sum(tf.multiply(inputs, weights), axis=-1)
        return outputs

    def discriminator(self, inputs, num_units, res_rate = 0.2, scope='Disc/t'):
        """
        The discriminator to score the distribution of time and event
        If the time is consistent with the history times, give high score.
        If it is on the constant, give low score.
        Implementation:
        CNN
        :param inputs: A Tensor. (N, L)
        :param num_units: An int
        :return A Tensor. (N, 1)"""
        with tf.compat.v1.variable_scope(scope, reuse=True):
            outputs = tf.expand_dims(inputs, axis=2)
            outputs = conv1d(outputs, num_units, scope='Disc/Conv1d')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_1')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_2')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_3')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_4')
            outputs = res_block(outputs, num_units, res_rate, scope='Disc/ResBlock_5')

            # if the output size is 1, it is the discriminator score of D
            # if the output size is 2, it is a bi-classification result of D
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(outputs)
            outputs = tf.squeeze(outputs, axis=-1)
            logits = tf.keras.layers.Dense(1)(outputs)
            # logging.info('The shape of output from D {}'.format(output.get_shape()))
            return logits


    def params_with_name(self, name):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return [v for v in variables if name in v.name]


    def event_loss(self, outputs_e, target_e):
        """
        :param outputs_e: A Tensor. (N, T, vocab_size)
        :param target_e: A tensor. (N, T)
        :return:
        """
        # Entropy for event sequence
        target_e = tf.one_hot(target_e, depth=outputs_e.get_shape()[-1])
        cross_entropy_loss = tf.losses.softmax_cross_entropy(logits=outputs_e,
                                                             onehot_labels=target_e,
                                                             scope='SeqLoss_e')
        return tf.reduce_mean(cross_entropy_loss)


    def time_loss(self, outputs_t, target_t):
        """
        :param outputs_t: A Tensor. (N, len)
        :param target_t: A Tensor. ()
        :return:
        """
        # Huber loss for time sequence
        huber_loss = tf.compat.v1.losses.huber_loss(labels=target_t,
                                                    predictions=outputs_t,
                                                    scope='HuberLoss_t')
        huber_loss = tf.reduce_mean(huber_loss)
        return huber_loss


    def loss_with_wasserstein(self, inputs_t, outputs_t, sample_t):
        """
        :param inputs_t: A Tensor. (N, T)
        :param outputs_t: A Tensor. (N, len)
        :param sample_t: A Tensor. (N, T+len)
        :return:
        """
        pred_t = tf.concat([inputs_t, outputs_t], axis=1)
        disc_fake = self.discriminator(pred_t, num_units=self.num_units)
        disc_real = self.discriminator(sample_t, self.num_units)

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


    def build(self):
        """
        :return:
        """
        # Embedding
        inputs_e, embedding_table  = embedding(self.inputs_e, self.vocab_size, self.num_units)
        # Encode events (N, T, C)
        outputs_e = self.Enc_e(inputs_e, self.num_units)  # (num_heads, N, T, C)
        # Encode times (N, T, C)
        inputs_t = tf.expand_dims(self.inputs_t, axis=2)
        outputs_t = self.Enc_t(inputs=inputs_t, num_units=self.num_units)
        # shared hidden representation
        hidden_e = tf.tile(tf.expand_dims(tf.reduce_sum(outputs_e, axis=1), axis=1), [1, self.len, 1])  # (N, len, C)
        hidden_t = tf.tile(tf.expand_dims(tf.reduce_sum(outputs_t, axis=1), axis=1), [1, self.len, 1])
        hidden_r = tf.concat([hidden_e, hidden_t], axis=-1)
        # Decode events
        outputs_e, logits_e = self.Gen_e(hidden_r, self.num_units, self.vocab_size)
        # Decode times
        if self.num_gen_t != 1:
            outputs_t, logits_t = self.M_Gen_t(hidden_r, self.num_units, self.num_gen_t)
            sel_weights, sel_index = self.selector(hidden_r, self.num_gen_t)
            l_logits_t = tf.reduce_sum(tf.multiply(logits_t, sel_weights), axis=-1)
            logits_t = self.output_t(logits_t, sel_index)
        else:
            outputs_t, logits_t = self.Gen_t(hidden_r, self.num_units)
            l_logits_t = logits_t
        disc_t_cost, gen_t_cost = self.loss_with_wasserstein(self.inputs_t, l_logits_t, self.sample_t)
        return logits_e, logits_t, l_logits_t, disc_t_cost, gen_t_cost


    def train_MLE(self, lr):
        MLE_params = self.params_with_name('Pred')
        # print('---gen e params')
        # print(gen_e_params)
        event_cross_entropy = self.event_loss(self.logits_e, self.target_e)
        time_huber_loss = self.time_loss(self.l_logits_t, self.target_t)
        MLE_loss = event_cross_entropy + time_huber_loss
        train_MLE_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(MLE_loss,
                                                                              var_list=MLE_params)
        return train_MLE_op


    def train_GAN(self, lr):
        # time generator params
        gen_t_params = self.params_with_name('Time')
        # print(gen_params)
        disc_t_params = self.params_with_name('Disc')
        # print(disc_params)
        gen_loss = self.gen_t_loss
        disc_loss = self.disc_t_loss
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(gen_loss, var_list=gen_t_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(disc_loss, var_list=disc_t_params)

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


    def cal_metrics(self, metric_k):
        # Define Metrics
        metric_k= metric_k
        # MRR@k
        precision, precision_op = tf.metrics.precision_at_k(labels=self.target_e,
                                                            predictions=self.logits_e,
                                                            k=metric_k, name='precision_k')
        # Isolate the variables stored behind the scenes by the metric operation
        running_precision_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="precision_k")
        # Define initializer to initialize/reset running variables
        running_precision_vars_initializer = tf.variables_initializer(var_list=running_precision_vars)

        # Recall@k
        recall, recall_op = tf.metrics.recall_at_k(labels=self.target_e,
                                                   predictions=self.logits_e,
                                                   k=metric_k,
                                                   name='recall_k')
        # Isolate the variables stored behind the scenes by the metric operation
        running_recall_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="recall_k")
        # Define initializer to initialize/reset running variables
        running_recall_vars_initializer = tf.variables_initializer(var_list=running_recall_vars)

        # MAE
        MAE, MAE_op = tf.metrics.mean_absolute_error(labels=self.target_t,
                                                     predictions=self.logits_t,
                                                     name='MAE')
        # Isolate the variables stored behind the scenes by the metric operation
        running_MAE_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="MAE")
        # Define initializer to initialize/reset running variables
        running_MAE_vars_initializer = tf.variables_initializer(var_list=running_MAE_vars)

        return MAE, MAE_op, running_MAE_vars_initializer,\
               precision, precision_op, running_precision_vars_initializer,\
               recall, recall_op, running_recall_vars_initializer