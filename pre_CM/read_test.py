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

"""Tests for models.tutorials.rnn.ptb.reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

import reader_event_sequence as reader
from classification_model import ClassficationModel as CM_model
from classification_model import CM_input
import CM_config as model_config 


class PtbReaderTest(tf.test.TestCase):

  def setUp(self):
    self._string_data = "\n".join(
        [" hello there i am",
         " rain as day",
         " want some cheesy puffs ?"])

  def get_config():
    """get the configuration of models """
    config = None
    if FLAGS.model == "small":
      config = model_config.SmallConfig()
    else:
      raise ValueError("Invalid model config: %s", FLAGS.model)
    if FLAGS.rnn_mode:
      config.rnn_mode = FLAGS.rnn_mode
    else:
      config.rnn_mode = BASIC
    return config

  def testPtbRawData(self):
    tmpdir = "/home/linli/data"
    input_file = "BPIC2012_cluster.txt"
    input_data_splited, inputWord2Id = reader.data_split(tmpdir, input_file, shuffle = True)
    cluster2Id = reader._build_cluster_dict(tmpdir, input_file)
    input_valid_traces, input_train_traces = reader.validation_training_data(
      input_data_splited, cluster2Id, inputWord2Id)
    input_test_traces = reader.test_data(
    input_data_splited, cluster2Id, inputWord2Id)
    config = get_config()
    train_input = CM_input(
        config = config,
        input_raw_traces = input_train_traces,
        name = "Train_input")
    with self.test_session() as sess:
      output = sess.run(train_input)
    self.assertEqual(len(train_input.input_data), 2)

  # def testPtbProducer(self):
  #   raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
  #   batch_size = 3
  #   num_steps = 2
  #   x, y = reader.ptb_producer(raw_data, batch_size, num_steps)
  #   with self.test_session() as session:
  #     coord = tf.train.Coordinator()
  #     tf.train.start_queue_runners(session, coord=coord)
  #     try:
  #       xval, yval = session.run([x, y])
  #       self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
  #       self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
  #       xval, yval = session.run([x, y])
  #       self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
  #       self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
  #     finally:
  #       coord.request_stop()
  #       coord.join()


if __name__ == "__main__":
  tf.test.main()