from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import random
import numpy
import datetime

def _get_cluster_number(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if ':' in line:
        cluster_number = len(line.split(':')[0].split('-'))
        break
  return cluster_number

def _read_cluster(filename):
  print("----------------------read----------------------")
  start_time = datetime.datetime.now()
  cluster_number = _get_cluster_number(filename)
  clusters = [[] for i in range(cluster_number)]
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if ':' in line:
        cluster = line.split(":")[0].split('-')
        for i, each_cluster in enumerate(cluster):
          clusters[i].append(each_cluster)
  end_time = datetime.datetime.now()
  print("time: " + str((end_time - start_time).seconds))
  return clusters

def _build_cluster_dict(filename):
  cluster_number = _get_cluster_number(filename)
  data = _read_cluster(filename)
  cluster2Id = []
  start_time = datetime.datetime.now()
  print("----------------------build----------------------")
  for each_cluster_list in data:
    counter = collections.Counter(each_cluster_list)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    cluster_to_id = dict(zip(words, range(len(words))))
    cluster2Id.append(cluster_to_id)
  end_time = datetime.datetime.now()
  print("time: " + str((end_time - start_time).seconds))
  return cluster2Id

def _get_attribute_number(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
      if ':' in line:
        attribute_number = len(line.split(':')[1].split()[0].split('-'))
        break
  return attribute_number

def _read_words(filename):
  print('--------------------read words--------------------')
  start_time = datetime.datetime.now()
  attribute_number = _get_attribute_number(filename)
  event_words = ['' for i in range(attribute_number)]
  with open(filename, "r") as f:
    lines = f.readlines()
    length = len(lines)
    for j,line in enumerate(lines):
      if j % ( length // 10 ) == 10:
        print(j * 1.0 / length)
      if ':' in line:
        line = [event for event in line.split(':')[1].split()]
        for event in line:
          for i, attr in enumerate(event.split('-')):
            if attr != '\n':
              string = attr + ' '
              event_words[i] += string
    end_time = datetime.datetime.now()
    print("time: " + str((end_time - start_time).seconds))
    return event_words


def _build_vocab(filename):
  data = _read_words(filename)
  word2Id = []
  start_time = datetime.datetime.now()
  print('-------------------build words--------------------')
  for event_attr_string in data:
    counter = collections.Counter(event_attr_string.split())
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    word2Id.append(word_to_id)
  end_time = datetime.datetime.now()
  print("time: " + str((end_time - start_time).seconds))
  return word2Id

# def readBatchOfSentences(filename, batchSize, minLen, startpos):
#   with open(filename, "r") as f:
#     sentences = f.readlines()
#     sentences = [ s.split() for s in sentences ]
#     t = [ (s[0:minLen], s[minLen:]) for s in sentences if len(s) > minLen + 1]
#     outp = numpy.empty( (batchSize, minLen), 'object' )
#     outs = numpy.empty( (batchSize), 'object' )
#     for c in range(startpos,startpos+batchSize):
#       outp[c - startpos] = t[c][0]
#       outs[c - startpos] = t[c][1]
#     return outp, outs

def write_dict(filename, dict):
  with open(filename, "w") as f:
    for word in dict.keys():
      f.write("{0} {1}\n".format(word, dict[word]))
    f.close()

def vocab_size(data_path=None, file=None):
  inputFName = os.path.join(data_path, file)
  word2Id = _build_vocab(inputFName)
  return [len(event_attr_dict) for event_attr_dict in word2Id]

# def _file_to_word_ids(filename, word_to_id):
#   data = _read_words(filename)
#   return [word_to_id[word] for word in data]

def build_vocab(dict_file):
  word2Id = []
  for i in range(3):
    word_to_id = dict()
    with open(dict_file.replace('-',str(i+1))) as f:
      lines = f.readlines()
      for line in lines:
        word_to_id[line.split(' ')[0]] = int(line.split(' ')[1])
    word2Id.append(word_to_id)
  return word2Id


def data_split(data_path=None, input_file=None, target_file=None, shuffle=True):
  num_pieces = 10
  inputFName = os.path.join(data_path, input_file) 
  targetFName = os.path.join(data_path, target_file)

  inputWord2Id = build_vocab(os.path.join(data_path, "event-_dict16.txt"))
  #targetWord2Id = _build_vocab(targetFName)
  targetWord2Id = inputWord2Id
  cluster2Id = _build_cluster_dict(inputFName)

  inputFile = open(inputFName, "r")
  inputTraces = inputFile.read().split('\n')
  inputFile.close()

  targetFile = open(targetFName, "r")
  targetTraces = targetFile.read().split('\n')
  targetFile.close()

  if (len(inputTraces) != len(targetTraces)):
      return None, None, None

  if (shuffle):
    indices = numpy.random.permutation(len(inputTraces))
    shuffledInputTraces = numpy.empty(len(inputTraces), dtype=numpy.object)
    numpy.put(shuffledInputTraces, indices, inputTraces)
    shuffledTargetTraces = numpy.empty(len(targetTraces), dtype=numpy.object)
    numpy.put(shuffledTargetTraces, indices, targetTraces)
  else:
    shuffledInputTraces = inputTraces
    shuffledTargetTraces = targetTraces

  nrow = len(shuffledInputTraces)
  data_size = nrow // num_pieces

  input_data_splited = [None] * 3
  target_data_splited = [None] * 3

  input_data_splited[0] = shuffledInputTraces[0:7*data_size]
  target_data_splited[0] = shuffledTargetTraces[0:7*data_size]

  input_data_splited[1] = shuffledInputTraces[7*data_size:9*data_size]
  target_data_splited[1] = shuffledTargetTraces[7*data_size:9*data_size]

  input_data_splited[2] = shuffledInputTraces[9*data_size:]
  target_data_splited[2] = shuffledTargetTraces[9*data_size:]

  inputVocabulary = len(inputWord2Id)
  targetVocabulary = len(targetWord2Id)

  return input_data_splited, target_data_splited, inputWord2Id, targetWord2Id, inputVocabulary, targetVocabulary, cluster2Id

class trace_object:
  def __init__(self):
    self.cluster = []
    self.event = []

def extract_cluster_event(trace_list, cluster2Id, event2Id):
  trace_extracted = []
  cluster_number = len(cluster2Id)
  event_attr_number = len(event2Id)
  
  for trace in trace_list:
    event_attrs = [[] for i in range(event_attr_number)]
    cluster_list = trace.split(':')[0].split('-')
    event_list = trace.split(':')[1]
    if ':' in trace:
      trace_item = trace_object()
      cluster_names = cluster_list
      event_words = event_list.split(' ')
      event_content = [each_event.split('-') for each_event in event_words]
      for i, word in enumerate(cluster_names):
        trace_item.cluster.append(cluster2Id[i][word])
      for word in event_content:
        for i, attr in enumerate(word):
          event_attrs[i].append(event2Id[i][attr.replace('\r','')]) 
      trace_item.event = event_attrs
    trace_extracted.append(trace_item)
  return trace_extracted

def validation_training_data(input_data_splited, target_data_splited, inputWord2Id, targetWord2Id, cluster2Id):
  input_validation_data = input_data_splited[1]
  input_validation_traces = extract_cluster_event(input_validation_data, cluster2Id, inputWord2Id)
  target_validation_data = target_data_splited[1]
  target_validation_traces = extract_cluster_event(target_validation_data, cluster2Id, targetWord2Id)

  input_training_data = input_data_splited[0]
  input_training_traces = extract_cluster_event(input_training_data, cluster2Id, inputWord2Id)
  target_training_data = target_data_splited[0]
  target_training_traces = extract_cluster_event(target_training_data, cluster2Id, targetWord2Id)

  return input_validation_traces, target_validation_traces, input_training_traces, target_training_traces

def test_data(input_data_splited, target_data_splited, inputWord2Id, targetWord2Id, cluster2Id):
  input_test_data = input_data_splited[2]
  input_test_traces = extract_cluster_event(input_test_data, cluster2Id, inputWord2Id)
  target_test_data = target_data_splited[2]
  target_test_traces = extract_cluster_event(target_test_data, cluster2Id, targetWord2Id)

  return input_test_traces, target_test_traces

def words_iterator(input_raw_traces, target_raw_traces, batch_size, num_steps):

  input_len_sum = 0
  cluster_data = []
  input_data = []
  target_data = []
  cluster_number = len(input_raw_traces[0].cluster)
  event_attr_number = len(input_raw_traces[0].event)

  print("length_traces: " + str(len(input_raw_traces))) #9320

  for (input_trace, target_trace) in zip(input_raw_traces, target_raw_traces):
    input_cluster_list = input_trace.cluster
    input_event_list = input_trace.event
    target_event_list = target_trace.event

    input_clusters = numpy.array(input_cluster_list, dtype = numpy.int32)
    input_events = numpy.array(input_event_list, dtype = numpy.int32)
    target_events = numpy.array(target_event_list, dtype = numpy.int32)
 
    data_len = input_events.shape[1]

    input_len = data_len - num_steps
    if input_len > 1:
      for i in range(input_len):
        input_data_element = input_events[:, i : i + num_steps]
        target_data_element = input_events[:, i +1 : i+1+ num_steps]
        cluster_element = input_clusters
        cluster_data.append(cluster_element)
        input_data.append(input_data_element)
        target_data.append(target_data_element)
      input_len_sum += input_len

  batch_len = input_len_sum // batch_size

  ziped_data = zip(cluster_data, input_data, target_data)

  for i in range(batch_len):
    c = numpy.zeros([cluster_number, batch_size])
    x = numpy.zeros([event_attr_number, batch_size, num_steps])
    y = numpy.zeros([event_attr_number, batch_size, num_steps])
    # WHY: Do we have to multiply by num_steps here? Why not let them overlap?
    feed_data = ziped_data[i*batch_size : (i+1)*batch_size]
    for j in range(batch_size):
      c[:,j] = feed_data[j][0]
      x[:,j] = feed_data[j][1]
      y[:,j] = feed_data[j][2]

    yield (c, x, y)

def read_sentences(input_raw_traces, minLen):
  t = []
  for input_trace in input_raw_traces:
    input_cluster_list = input_trace.cluster
    input_event_list = input_trace.event

    if len(input_event_list[0]) <= minLen + 1:
      continue

    input_clusters = numpy.array(input_cluster_list, dtype = numpy.int32)
    input_events = numpy.array(input_event_list, dtype = numpy.int32)

    t.append([input_clusters, input_events[:,0:minLen], input_events[:,minLen:]])

  return t

def read_batch_of_sentences(input_trace_list, batchSize, minLen, startpos):
  t = input_trace_list
  
  c_list = []
  p_list = []
  s_list = []
  print("cluster num: " + str(len(t[0][0])))
  print("event num: " + str(len(t[0][1])))

  for i in range(len(t[0][0])):
    #print("C: " + str(i))
    outc = numpy.empty( (batchSize), 'object' )
    for c in range(startpos, startpos+batchSize):
      outc[c - startpos] = t[c][0][i]
    c_list.append(outc)
  for i in range(len(t[0][1])):
    #print("p\s: " + str(i))
    outp = numpy.empty( (batchSize, minLen), 'object' )
    outs = numpy.empty( (batchSize), 'object' )
    for c in range(startpos, startpos+batchSize):
      outp[c - startpos] = t[c][1][i]
      outs[c - startpos] = t[c][2][i]
    p_list.append(outp)
    s_list.append(outs)
  #print(s_list)
  return c_list, p_list, s_list