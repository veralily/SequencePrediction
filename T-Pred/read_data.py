from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random


def data_split(event_file=None, time_file=None, shuffle=True):
    num_pieces = 10
    read_event = open(event_file, "r")
    event_traces = read_event.read().split('\n')
    read_event.close()

    read_time = open(time_file, "r")
    time_traces = read_time.read().split('\n')
    read_time.close()

    if len(event_traces) != len(time_traces):
        print('number of event traces and time traces are not equal!!')
        return None, None, None
    else:
        nrow = len(event_traces)
        print('length of traces: ', len(event_traces), len(time_traces))

    data_size = nrow // num_pieces
    test_event_traces = event_traces[:data_size]
    test_time_traces = time_traces[:data_size]
    test_data = zip(test_event_traces, test_time_traces)

    data4train = zip(event_traces[data_size:], time_traces[data_size:])

    if shuffle:
        np.random.shuffle(data4train)
    else:
        pass

    train_data = data4train[:7 * data_size]
    valid_data = data4train[7 * data_size:-1]

    '''
	The splited data has shape [2, num_of_traces, length]
	'''

    return train_data, valid_data, test_data


def data_iterator(input_data, num_steps, length, overlap=True):
    input_len_sum = 0
    input_event_data = []
    target_event_data = []
    input_time_data = []
    target_time_data = []
    print("length of traces: " + str(len(input_data)))  # 9320

    for event_trace, time_trace in input_data:
        events = np.array([int(event) for event in event_trace.split('\t')], dtype=np.int32)
        times = np.array([float(time) for time in time_trace.split('\t')], dtype=np.float32)
        data_len = len(events)
        time_len = len(times)
        # print('length of items in event and time sequence: ', data_len, time_len)

        if data_len != time_len:
            print("The length of data traces and that of time traces are noe equal!")

        if overlap:
            input_len = data_len - (num_steps + length)
            if input_len > 1:
                for i in range(input_len):
                    input_event_elem = events[i: i + num_steps]
                    target_event_elem = events[i + num_steps: i + num_steps + length]
                    input_time_elem = times[i: i + num_steps]
                    target_time_elem = times[i + num_steps: i + num_steps + length]
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
                    input_event_elem = events[i * seg_length: i * seg_length + num_steps]
                    target_event_elem = events[i * seg_length + num_steps: (i + 1) * seg_length]
                    input_time_elem = times[i * seg_length: i * seg_length + num_steps]
                    target_time_elem = times[i * seg_length + num_steps: (i + 1) * seg_length]
                    input_event_data.append(input_event_elem)
                    target_event_data.append(target_event_elem)
                    input_time_data.append(input_time_elem)
                    target_time_data.append(target_time_elem)
                input_len_sum += input_len
            else:
                pass
            # print("the trace is too short")

    return input_event_data, target_event_data, input_time_data, target_time_data


def generate_batch(batch_size, input_event_data, target_event_data, input_time_data, target_time_data):
    batch_num = len(input_event_data) // batch_size
    for i in range(batch_num):
        e_x = input_event_data[i * batch_size:(i + 1) * batch_size]
        e_y = target_event_data[i * batch_size:(i + 1) * batch_size]
        t_x = input_time_data[i * batch_size:(i + 1) * batch_size]
        t_y = target_time_data[i * batch_size:(i + 1) * batch_size]
        yield e_x, e_y, t_x, t_y


def generate_sample_t(batch_size, input_time_data, target_time_data):
    t_list = np.concatenate([input_time_data, target_time_data], axis=1)
    return random.sample(list(t_list), batch_size)


def batch_count(input_data, num_steps, length, batch_size, overlap=True):
    input_len_sum = 0
    for event_trace, time_trace in input_data:
        if event_trace == '':
            print('---------null trace--------')
            events = times = None
        else:
            events = np.array([int(event) for event in event_trace.split('\t')], dtype=np.int32)
            times = np.array([float(time) for time in time_trace.split('\t')], dtype=np.float32)
        data_len = len(events)
        time_len = len(times)

        if data_len != time_len:
            print("The length of data traces and that of time traces are noe equal!")

        if overlap:
            input_len = data_len - (num_steps + length)
            if input_len > 1:
                input_len_sum += input_len
        else:
            input_len = (data_len - 1) // (num_steps + length)
            if input_len > 0:
                input_len_sum += input_len
    return input_len_sum // batch_size
