
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import numpy
import datetime
import tensorflow as tf

def _get_attribute_number(data_path):
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if ':' in line:
        attribute_number = len(line.split(':')[1].split()[0].split('-'))
        break
  return attribute_number

def _read_words(filename):
  print('--------------------read words--------------------')
  with tf.gfile.GFile(filename, 'r') as f:
    return f.read().decode('utf-8').replace('\n',' ').split()

def _build_vocab(filename):
  data = _read_words(filename)
  print('-------------------build words--------------------')

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word2Id = dict(zip(words, range(len(words))))

  return word2Id

def _trace_to_word_ids(trace, word2Id):
  return [word2Id[word] for word in trace if word in word2Id]

def write_dict(filename, dict):
  with open(filename, "w") as f:
    for word in dict.keys():
      f.write("{0} {1}\n".format(word, dict[word]))
    f.close()

def vocab_size(data_path=None, file=None):
  inputFName = os.path.join(data_path, file)
  word2Id = _build_vocab(inputFName)
  return len(word2Id)

def data_split(input_traces, shuffle=True):
  num_pieces = 10

  if (shuffle):
    numpy.random.shuffle(input_traces)
    shuffled_input_traces = input_traces
  else:
    shuffled_input_traces = input_traces

  nrow = len(shuffled_input_traces)
  data_size = nrow // num_pieces

  input_data_splited = [None] * 3

  input_data_splited[0] = shuffled_input_traces[0:7*data_size]
  input_data_splited[1] = shuffled_input_traces[7*data_size:9*data_size]
  input_data_splited[2] = shuffled_input_traces[9*data_size:]

  return input_data_splited[0], input_data_splited[1], input_data_splited[2]

def raw_data(data_path = None):
  event_data_path = os.path.join(data_path, "BPIC2014_event.txt")
  label_path = os.path.join(data_path, "BPIC2014_cluster2.txt")

  word2Id = _build_vocab(event_data_path)
  label2Id = _build_vocab(label_path)
  print(label2Id)

  with tf.gfile.GFile(event_data_path, 'r') as f:
    event_data_traces = f.read().decode('utf-8').split('\n')
    event_data = [_trace_to_word_ids(trace.split(' '), word2Id) for trace in event_data_traces]

  with tf.gfile.GFile(label_path, 'r') as f:
    label_traces = f.read().decode('utf-8').split('\n')
    labels = [label2Id[label] for label in label_traces]

  print("-------length of event traces : %d ----------" % len(event_data))
  print("-------number of laebels: %d ----------" % len(labels))

  data_zip = zip(event_data, labels)

  if len(event_data) != len(labels):
    print("The num of trace is not equal!")
  else:
    return data_split(data_zip)

def data_clip(input_raw_traces, num_steps, batch_size, overlap = True):
  input_len_sum = 0
  labels_data = []
  input_data = []
  print("length_traces: " + str(len(input_raw_traces))) #9320

  for event_trace, label in list(input_raw_traces):
    input_events = numpy.array(event_trace, dtype = numpy.int32)
    data_len = len(input_events)

    if (overlap):
      input_len = data_len - num_steps
      if input_len > 1:
        for i in range(input_len):
          input_data_element = input_events[i : i + num_steps]
          labels_data.append(label)
          input_data.append(input_data_element)
        input_len_sum += input_len
      else:
        pass
      	# print("the trace is too short")
    else:
      input_len = (data_len - 1) // num_steps
      if input_len > 0:
      	for i in range(input_len):
      	  input_data_element = input_events[i * num_steps : (i + 1)*num_steps]
          labels_data.append(label)
          input_data.append(input_data_element)
        input_len_sum += input_len
      else:
        pass
      	# print("the trace is too short")

  batch_len = input_len_sum // batch_size
  return batch_len, numpy.array(input_data, dtype = numpy.int32), numpy.array(labels_data, dtype = numpy.int32)

def data_producer(input_raw_data, labels_raw_data, batch_len, batch_size, num_steps, name):

  with tf.name_scope(name, "DataProducer", [input_raw_data, labels_raw_data, batch_len, batch_size, num_steps]):
    labels_data = tf.convert_to_tensor(labels_raw_data, name="raw_label_data", dtype=tf.int32)
    input_data = tf.convert_to_tensor(input_raw_data, name="raw_input_data", dtype=tf.int32)

    print("batch_len: %d" % batch_len)
    print("batch_size: %d" % batch_size)

    labels_data = tf.reshape(labels_data[0: batch_len * batch_size], [batch_len, batch_size])
    input_data = tf.reshape(input_data[0 : batch_size * batch_len,:],
                      [batch_len, batch_size, input_data.get_shape()[1]])

    print(labels_data.get_shape())
    print(input_data.get_shape())

    epoch_size = batch_len

    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    print(i)
    x = input_data[i]     
    c = labels_data[i]
    return x, c
