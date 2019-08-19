from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import random
import numpy as np
import datetime
from tqdm import tqdm


def read_words(filename):
	print('----read words----')
	start_time = datetime.datetime.now()
	event_words = []
	with open(filename, "r") as f:
		lines = f.readlines()
		length = len(lines)
		for i in tqdm(range(length)):
			if lines[i] != '':
				line = lines[i].replace('\n','').split('\t')
				for event in line:
					event_words.append(event)
		end_time = datetime.datetime.now()
		print("time: " + str((end_time - start_time).seconds))
		return event_words


def build_vocab(filename):
	data = read_words(filename)
	start_time = datetime.datetime.now()
	print('----build words----')
	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
	words, _ = list(zip(*count_pairs))
	word_to_id = dict(zip(words, range(len(words))))
	end_time = datetime.datetime.now()
	print("time: " + str((end_time - start_time).seconds))
	print("event_dict length: " + str(len(word_to_id)))
	return word_to_id


def write_dict(filename, dict):
	with open(filename, "w") as f:
		for word in dict.keys():
			f.write("{0} {1}\n".format(word, dict[word]))
		f.close()


def vocab_size(filenamee):
	word_to_id = build_vocab(filenamee)
	return len(word_to_id)


def data_split(event_file=None, time_file=None, shuffle=True):
	num_pieces = 10
	read_event = open(event_file, "r")
	event_traces = read_event.read().split('\n')
	read_event.close()

	read_time = open(time_file, "r")
	time_traces = read_time.read().split('\n')
	read_time.close()

	if (len(event_traces) != len(time_traces)):
		return None, None, None

	data_zip = zip(event_traces, time_traces)
	
	if (shuffle):
		np.random.shuffle(data_zip)
		shuffled_input_traces = data_zip
	else:
		shuffled_input_traces = data_zip

	nrow = len(shuffled_input_traces)
	data_size = nrow // num_pieces

	train_data = shuffled_input_traces[0:7*data_size]
	valid_data = shuffled_input_traces[7*data_size:9*data_size]
	test_data = shuffled_input_traces[9*data_size:-1]

	return train_data, valid_data, test_data


def data_iterator(input_raw_traces, event_to_id, num_steps, length, overlap = True):
	input_len_sum = 0
	input_event_data = []
	target_event_data = []
	input_time_data = []
	target_time_data = []
	print("length_traces: " + str(len(input_raw_traces))) #9320

	for event_trace, time_trace in list(input_raw_traces):
		if event_trace == '':
			pass
		else:
			events = np.array([event_to_id[event] for event in event_trace.split('\t')], dtype =np.int32)
			times = np.array([-1.0 if time == time_trace.split('\t')[-1] else float(time) for time in time_trace.split('\t')], dtype = np.float32)
		data_len = len(events)
		time_len = len(times)

		if (overlap):
			input_len = data_len - (num_steps + length)
			if input_len > 1:
				for i in range(input_len):
					input_event_elem = events[i : i + num_steps]
					target_event_elem = events[i+num_steps: i+num_steps+length]
					input_time_elem = times[i : i + num_steps]
					target_time_elem = times[i+num_steps : i+num_steps+length]
					input_event_data.append(input_event_elem)
					target_event_data.append(target_event_elem)
					input_time_data.append(input_time_elem)
					target_time_data.append(target_time_elem)
				input_len_sum += input_len
			else:
				pass
				# print("the trace is too short")
		else:
			input_len = (data_len - 1) // (num_steps + length)
			seg_length = num_steps + length
			if input_len > 0:
				for i in range(input_len):
					input_event_elem = events[i*seg_length : i*seg_length + num_steps]
					target_event_elem = events[i*seg_length+num_steps : (i+1)*seg_length]
					input_time_elem = times[i*seg_length : i*seg_length + num_steps]
					target_time_elem = times[i*seg_length+num_steps: (i+1)*seg_length]
					input_event_data.append(input_event_elem)
					target_event_data.append(target_event_elem)
					input_time_data.append(input_time_elem)
					target_time_data.append(target_time_elem)
				input_len_sum += input_len
			else:
				pass
				# print("the trace is too short")

	input_event_data = np.array(input_event_data, dtype = np.int32)
	target_event_data = np.array(target_event_data, dtype = np.int32)
	input_time_data = np.array(input_time_data, dtype = np.float32)
	target_time_data = np.array(target_time_data, dtype = np.float32)

	return input_len_sum, input_event_data, target_event_data, input_time_data, target_time_data


def generate_batch(input_len, batch_size, input_event_data, target_event_data, input_time_data, target_time_data):

	batch_num = input_len // batch_size

	e_x_list = []
	e_y_list = []
	t_x_list = []
	t_y_list = []

	for i in range(batch_num):
		e_x_list.append(input_event_data[i*batch_size:(i+1)*batch_size,:])
		e_y_list.append(target_event_data[i*batch_size:(i+1)*batch_size,:])
		t_x_list.append(input_time_data[i*batch_size:(i+1)*batch_size,:])
		t_y_list.append(target_time_data[i*batch_size:(i+1)*batch_size,:])

	return batch_num, e_x_list, e_y_list, t_x_list, t_y_list


def generate_sample_t(input_len, batch_size, input_time_data, target_time_data):

	batch_num = input_len // batch_size

	t_sample_list = []
	t_list = np.concatenate([input_time_data, target_time_data], axis=1)
	np.random.shuffle(t_list)

	for i in range(batch_num):
		t_sample_list.append(t_list[i*batch_size : (i+1)*batch_size, :])

	return batch_num, t_sample_list


def batch_count(input_raw_traces, num_steps, length, batch_size, overlap=True):
	input_len_sum = 0
	for event_trace, time_trace in list(input_raw_traces):
		if event_trace == '':
			print('---------null trace--------')
		else:
			events = event_trace.split('\t')
			times = np.array([float(time) if time != '[EOS]' else 0.0 for time in time_trace.split('\t')], dtype = np.float32)
		data_len = len(events)
		time_len = len(times)

		if (overlap):
			input_len = data_len - (num_steps + length)
			if input_len > 1:
				input_len_sum += input_len
		else:
			input_len = (data_len - 1) // (num_steps + length)
			if input_len > 0:
				input_len_sum += input_len
	return input_len_sum // batch_size

