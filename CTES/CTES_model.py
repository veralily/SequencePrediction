# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
- rnn_mode - the low level implementation of lstm cell: one of CUDNN,
             BASIC, or BLOCK, representing cudnn_lstm, basic_lstm, and
             lstm_block_cell classes.
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader_event_sequence as reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/home/linli/data/BPIC2017",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", "./Model17/",
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_integer("num_gpus", 3,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_string("rnn_mode", None,
                    "The low level implementation of lstm cell: one of CUDNN, "
                    "BASIC, and BLOCK, representing cudnn_lstm, basic_lstm, "
                    "and lstm_block_cell classes.")
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class CTESAInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size, self.input_event_data, self.cluster_data,self.input_attr_data, self.target_event_data, self.target_attr_data = reader.data_clip(
      data, num_steps, batch_size, overlap = True)
    read_content = reader.data_producer(
      self.input_event_data,
      self.cluster_data,
      self.input_attr_data,
      self.target_event_data,
      self.target_attr_data,
      self.epoch_size,
      self.batch_size,
      self.num_steps,
      name = name)

    self.input_e_seq, self.target_e_seq, self.cluster_list, self.attr_seq_list, self.target_attr_list = read_content


class CTESAModel(object):
  """The classification model."""

  def __init__(self, is_training, config, input_):
    self._is_training = is_training
    self._input = input_
    self._rnn_params = None
    self._cell = None
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size
    label_size = config.label_size
    attr_size = config.attr_size

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs_event = tf.nn.embedding_lookup(embedding, input_.input_e_seq)

    if is_training and config.keep_prob < 1:
      inputs_event = tf.nn.dropout(inputs_event, config.keep_prob)

    outputs_event, state_event = self._build_rnn_graph(inputs_event, config, is_training, "RNN_event")

    inputs_attr_1 = input_.attr_seq_list[0]
    inputs_attr_1 = tf.one_hot(inputs_attr_1, depth = attr_size[0])
    outputs_attr_1, state_attr_1 = self._build_rnn_graph(inputs_attr_1, config, is_training, "RNN_attr1")

    inputs_attr_2 = input_.attr_seq_list[1]
    inputs_attr_2 = tf.one_hot(inputs_attr_2, depth = attr_size[1])
    outputs_attr_2, state_attr_2 = self._build_rnn_graph(inputs_attr_2, config, is_training, "RNN_attr2")

    inputs_attr_3 = input_.attr_seq_list[2]
    inputs_attr_3 = tf.one_hot(inputs_attr_3, depth = attr_size[2])
    outputs_attr_3, state_attr_3 = self._build_rnn_graph(inputs_attr_3, config, is_training, "RNN_attr3")

    ### compute the correlation between e and a
    outputs_z = []
    variable_b = []
    z = tf.concat([tf.multiply(outputs_event[-1], outputs_attr_1[-1]), outputs_event[-1], outputs_attr_1[-1], outputs_attr_2[-1], outputs_attr_3[-1]], 1)
    with tf.variable_scope("modulator"):
        z_w = tf.get_variable("z_w", [z.get_shape()[1], 4], dtype=data_type())
        z_b = tf.get_variable("z_b", [4], dtype = data_type())
        logits_z = tf.nn.xw_plus_b(z, z_w, z_b)
        b = tf.sigmoid(logits_z)
    for i in range(config.num_steps):
      output_event = tf.expand_dims(outputs_event[i], -1)
      output_attr_1 = tf.expand_dims(outputs_attr_1[i], -1)
      output_attr_2 = tf.expand_dims(outputs_attr_2[i], -1)
      output_attr_3 = tf.expand_dims(outputs_attr_3[i], -1)
      input_z = tf.reshape(tf.transpose(tf.concat([output_event,output_attr_1,output_attr_2,output_attr_3], -1), [1,0,2]), [output_event.get_shape()[1], -1])
      output_z = tf.transpose(tf.reshape(tf.multiply(tf.reshape(b, [-1]), input_z), [output_event.get_shape()[1], self.batch_size, -1]),[1,0,2])
      output_z = tf.reduce_sum(output_z, 2)
      outputs_z.append(output_z)
      variable_b.append(tf.expand_dims(tf.reduce_mean(b,0),0))

    self._b = tf.reduce_mean(tf.concat(variable_b, 0),0)


    outputs = tf.transpose(tf.convert_to_tensor(outputs_z, dtype = data_type()), [1,0,2])
    outputs, state_event = self._build_rnn_graph(outputs, config, is_training, "RNN2")

    ### create  predicted seq
    output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    softmax_w = tf.get_variable(
      "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

    # Use the contrib sequence loss and average over the batches
    loss_seq = tf.contrib.seq2seq.sequence_loss(
        logits,
        input_.target_e_seq,
        tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=True,
        average_across_batch=False)
    correct_prediction = tf.equal(tf.argmax(tf.reshape(logits,[-1, vocab_size]), 1), tf.cast(tf.reshape(input_.target_e_seq, [-1]), tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ### predict the class of seq
    # output = tf.reshape(tf.concat(outputs, 1), [self.batch_size, -1])
    # print("----------output for class-----------")
    # print(output.get_shape())
    # softmax_w_class = tf.get_variable(
    #     "softmax_w_class", [size * self.num_steps, label_size], dtype=data_type())
    # softmax_b_class = tf.get_variable("softmax_b_class", [label_size], dtype=data_type())
    # logits_class = tf.nn.xw_plus_b(
    #   output,
    #   softmax_w_class,
    #   softmax_b_class)
    # # Reshape logits to be a 3-D tensor for sequence loss
    # logits_class = tf.reshape(logits_class, [self.batch_size, label_size])

    # ### the loss function of pre-experiment about classification of event sequences
    # loss_class  = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   labels=tf.cast(input_.cluster_list[1], tf.int64),
    #   logits=logits_class,
    #   name='cross_entropy_per_example')

    ### the sum of losses
    loss = loss_seq
    # Update the cost
    self._cost = tf.reduce_mean(loss)
    self._final_state_event = state_event
    self._accuracy = accuracy

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def _build_rnn_graph(self, inputs, config, is_training, variable_name_scope):
    if config.rnn_mode == CUDNN:
      return self._build_rnn_graph_cudnn(inputs, config, is_training, None)
    else:
      return self._build_rnn_graph_lstm(inputs, config, is_training, variable_name_scope)

  def _build_rnn_graph_cudnn(self, inputs, config, is_training, variable_name_scope):
    """Build the inference graph using CUDNN cell."""
    inputs = tf.transpose(inputs, [1, 0, 2])
    self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=config.num_layers,
        num_units=config.hidden_size,
        input_size=config.hidden_size,
        dropout=1 - config.keep_prob if is_training else 0)
    params_size_t = self._cell.params_size()
    self._rnn_params = tf.get_variable(
        "lstm_params",
        initializer=tf.random_uniform(
            [params_size_t], -config.init_scale, config.init_scale),
        validate_shape=False)
    c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                 tf.float32)
    self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
    outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, config.hidden_size])
    return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

  def _get_lstm_cell(self, config, is_training):
    if config.rnn_mode == BASIC:
      return tf.contrib.rnn.BasicLSTMCell(
          config.hidden_size, forget_bias=0.0, state_is_tuple=True,
          reuse=not is_training)
    if config.rnn_mode == BLOCK:
      return tf.contrib.rnn.LSTMBlockCell(
          config.hidden_size, forget_bias=0.0)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

  def _build_rnn_graph_lstm(self, inputs, config, is_training, variable_name_scope):
    """Build the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell():
      cell = self._get_lstm_cell(config, is_training)
      if is_training and config.keep_prob < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=config.keep_prob)
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(config.batch_size, data_type())
    state = self._initial_state
    # Simplified version of tf.nn.static_rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    # outputs, state = tf.nn.static_rnn(cell, inputs,
    #                                   initial_state=self._initial_state)
    outputs = []
    with tf.variable_scope(variable_name_scope):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    # output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
    # output = tf.reshape(cell_output, [config.batch_size, config.hidden_size])
    return outputs, state

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    ops = {util.with_prefix(self._name, "cost"): self._cost}
    ops.update(accuracy = self._accuracy)
    ops.update(b = self._b)
    if self._is_training:
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
      if self._rnn_params:
        ops.update(rnn_params=self._rnn_params)
    for name, op in ops.items():
      tf.add_to_collection(name, op)
    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_event_name = util.with_prefix(self._name, "final_state_event")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state_event, self._final_state_event_name)

  def import_ops(self):
    """Imports ops from collections."""
    if self._is_training:
      self._train_op = tf.get_collection_ref("train_op")[0]
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      rnn_params = tf.get_collection_ref("rnn_params")
      if self._cell and rnn_params:
        params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
            self._cell,
            self._cell.params_to_canonical,
            self._cell.canonical_to_params,
            rnn_params,
            base_variable_scope="Model/RNN")
        tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
    self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
    self._accuracy = tf.get_collection_ref("accuracy")[0]
    self._b = tf.get_collection_ref("b")[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state_event = util.import_state_tuples(
        self._final_state_event, self._final_state_event_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def accuracy(self):
    return self._accuracy

  @property
  def b(self):
    return self._b
  

  @property
  def final_state_event(self):
    return self._final_state_event

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_event_name(self):
    return self._final_state_event_name