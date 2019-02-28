
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
  print("------data_split_function------length of input traces: %d" %len(input_traces))
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

  file_list = os.listdir(data_path)

  attr_files = []
  cluster_files = []

  for filename in file_list:
    if "event" in filename:
      event_file = filename
    else:
      if "a" in filename:
        attr_files.append(filename)
      else:
        if "cluster" in filename:
          cluster_files.append(filename)

  ### get the event sequences 
  event_data_path = os.path.join(data_path, event_file)
  word2Id = _build_vocab(event_data_path)

  with tf.gfile.GFile(event_data_path, 'r') as f:
    event_data_traces = f.read().decode('utf-8').split('\n')
    event_data = [_trace_to_word_ids(trace.split(' '), word2Id) for trace in event_data_traces]

  ### get the cluster sequences
  cluster_data_list = []
  for filename in cluster_files:
    cluster_data_path = os.path.join(data_path, filename)
    cluster2Id = _build_vocab(cluster_data_path)
    with tf.gfile.GFile(cluster_data_path, 'r') as f:
      cluster_traces = f.read().decode('utf-8').split('\n')
      single_cluster_data = [cluster2Id[label] for label in cluster_traces]
      # print("***** a single cluster: %d " % len(single_cluster_data))
    cluster_data_list.append(single_cluster_data)
  cluster_data = zip(*cluster_data_list)

  ### get the attribute sequences
  attr_data_list = []
  for filename in attr_files:
    attr_data_path = os.path.join(data_path, filename)
    attr2Id = _build_vocab(attr_data_path)
    with tf.gfile.GFile(attr_data_path, 'r') as f:
      attr_traces = f.read().decode('utf-8').split('\n')
      single_attr_data = [_trace_to_word_ids(trace.split(' '), attr2Id) for trace in attr_traces]
      # print("***** a single attr trace: %d " % len(single_attr_data))
    attr_data_list.append(single_attr_data)
  attr_data = zip(*attr_data_list)

  print("-------length of event traces : %d ----------" % len(event_data))
  print("-------length of cluster traces : %d ----------" % len(cluster_data))
  print("-------length of attr traces : %d ----------" % len(attr_data))

  data_zip = zip(event_data, cluster_data, attr_data)

  print("-------length of data zipped : %d -----------" % len(data_zip))
  
  return data_split(data_zip)

def data_clip(input_raw_traces, num_steps, batch_size, overlap = True):
  input_len_sum = 0
  cluster_data = []
  input_event_data = []
  target_events = []
  target_attrs = []
  input_attr_data = []
  print("length_traces: " + str(len(input_raw_traces))) #9320

  for event_trace, clusters, attrs in list(input_raw_traces):
    input_events = numpy.array(event_trace, dtype = numpy.int32)
    input_attributes = numpy.array(attrs, dtype = numpy.int32)
    data_len = len(input_events)

    if (overlap):
      input_len = data_len - num_steps
      if input_len > 1:
        for i in range(input_len):
          input_data_element = input_events[i : i + num_steps]
          target_data_element = input_events[i + 1: i + num_steps + 1]
          input_attr_element = input_attributes[:, i:i + num_steps]
          target_attr_element = input_attributes[:, i+1: i+num_steps+1]
          cluster_data.append(list(clusters))
          input_event_data.append(input_data_element)
          target_events.append(target_data_element)
          input_attr_data.append(input_attr_element)
          target_attrs.append(target_attr_element)
        input_len_sum += input_len
      else:
        pass
        # print("the trace is too short")
    else:
      input_len = (data_len - 1) // num_steps
      if input_len > 0:
        for i in range(input_len):
          input_data_element = input_events[i * num_steps : (i + 1)*num_steps]
          target_data_element = input_events[i * num_steps + 1 : (i + 1)*num_steps + 1]
          input_attr_element = input_attributes[:, i * num_steps : (i + 1)*num_steps]
          target_attr_element = input_attributes[:, i * num_steps + 1 : (i + 1)*num_steps + 1]
          cluster_data.append(list(clusters))
          input_event_data.append(input_data_element)
          target_events.append(target_data_element)
          input_attr_data.append(input_attr_element)
          target_attrs.append(target_attr_element)
        input_len_sum += input_len
      else:
        pass
        # print("the trace is too short")

  batch_len = input_len_sum // batch_size
  input_event_data = numpy.array(input_event_data, dtype = numpy.int32)
  cluster_data = numpy.array(cluster_data, dtype = numpy.int32)
  input_attr_data = numpy.array(input_attr_data, dtype = numpy.int32)
  target_event_data = numpy.array(target_events, dtype = numpy.int32)
  target_attr_data = numpy.array(target_attrs, dtype = numpy.int32)

  return batch_len, input_event_data, cluster_data, input_attr_data, target_event_data, target_attr_data 

def data_producer(input_raw_data, cluster_raw_data, attr_raw_data, target_event_data, target_attr_data, batch_len, batch_size, num_steps, name = None):

  with tf.name_scope(name, "DataProducer", [input_raw_data, cluster_raw_data, attr_raw_data, target_event_data, target_attr_data, batch_len, batch_size, num_steps]):
    cluster_data = tf.convert_to_tensor(cluster_raw_data, name="raw_label_data", dtype=tf.int32)
    input_data = tf.convert_to_tensor(input_raw_data, name="raw_input_data", dtype=tf.int32)
    attr_data = tf.convert_to_tensor(attr_raw_data, name = "raw_attr_data", dtype=tf.int32)
    target_data = tf.convert_to_tensor(target_event_data, name = "target_event_data", dtype= tf.int32)
    target_attr = tf.convert_to_tensor(target_attr_data, name = "target_attr_data", dtype = tf.int32)

    print("batch_len: %d" % batch_len)
    print("batch_size: %d" % batch_size)

    
    input_data = tf.reshape(input_data[0 : batch_size * batch_len,:],
      [batch_len, batch_size, input_data.get_shape()[1]])
    target_data = tf.reshape(target_data[0 : batch_size * batch_len,:],
      [batch_len, batch_size,target_data.get_shape()[1]])
    attr_data = tf.reshape(attr_data[0:batch_len * batch_size,:,:],
      [batch_len, batch_size,attr_data.get_shape()[1],attr_data.get_shape()[2]])
    target_attr = tf.reshape(target_attr[0:batch_len * batch_size,:,:],
      [batch_len, batch_size,target_attr.get_shape()[1],target_attr.get_shape()[2]])
    cluster_data = tf.reshape(cluster_data[0: batch_len * batch_size, :],
      [batch_len, batch_size,cluster_data.get_shape()[1]])


    cluster_data = tf.transpose(cluster_data, [2,0,1])
    attr_data = tf.transpose(attr_data, [2,0,1,3])
    target_attr = tf.transpose(target_attr, [2,0,1,3])

    print(cluster_data.get_shape())
    print(input_data.get_shape())
    print(attr_data.get_shape())

    epoch_size = batch_len

    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    x = input_data[i]
    y = target_data[i]  
    c = []
    for t in range(cluster_data.get_shape()[0]):
      cluster = cluster_data[t]
      c.append(cluster[i])
    a = []
    for t in range(attr_data.get_shape()[0]):
      attr = attr_data[t]
      a.append(attr[i])
    t_a = []
    for t in range(target_attr.get_shape()[0]):
      t_attr = target_attr[t]
      t_a.append(t_attr[i])
  return x, y, c, a, t_a