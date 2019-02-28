import os
import numpy as np
import tensorflow as tf
from RNN_Cell import T_GRUCell
from RNN_Cell import GRUCell
from RNN_Cell import T_LSTMCell
from RNN_Cell import LSTMCell

_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def alias_params(replace_dict, param_aliases):
    for old,new in replace_dict.items():
        # print "aliasing {} to {}".format(old,new)
        param_aliases[old] = new

    return param_aliases

def delete_param_aliases(param_aliases):
    param_aliases.clear()
    return param_aliases


def build_rnn_graph(hidden_r, num_layers, hidden_size, batch_size, length, name):
	def make_cell():
		cell = tf.contrib.rnn.GRUBlockCellV2(
			num_units = hidden_size,
			reuse = None)
		return cell

	cell = tf.contrib.rnn.MultiRNNCell(
		[make_cell() for _ in range(num_layers)], state_is_tuple = True)

	outputs = []
	with tf.variable_scope(name):
		initial_state = cell.zero_state(batch_size, tf.float32)
		state = initial_state

		for time_step in range(length):
			if time_step > 0: tf.get_variable_scope().reuse_variables()
			(cell_output, state) = cell(hidden_r, state)
			outputs.append(cell_output)

	return outputs

def build_encoder_graph_t(cell_type, inputs, t, hidden_size, num_layers, batch_size, num_steps, keep_prob, is_training, name):
    def make_cell():
        if cell_type == 'T_GRUCell':
            cell = T_GRUCell(
                hidden_size,
                reuse = not is_training,
                name = 'T_GRUCell')
        if cell_type == 'T_LSTMCell':
            cell = T_LSTMCell(
                hidden_size,
                reuse = not is_training,
                name = 'T_LSTMCell')
        if is_training and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_pron = keep_prob)
        return cell

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple = True)
    
    inputs = tf.concat([inputs, tf.expand_dims(t,2)],2)
    outputs = []
    with tf.variable_scope(name):
        initial_state = cell.zero_state(batch_size, tf.float32)
        state = initial_state

        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

    return outputs

def build_encoder_graph_gru(inputs, hidden_size, num_layers, batch_size, num_steps, keep_prob, is_training, name):
    def make_cell():
        cell = GRUCell(
            hidden_size,
            reuse = not is_training,
            name = 'GRUCell')
        if is_training and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                output_keep_pron = keep_prob)
        return cell

    cell = tf.nn.rnn_cell.MultiRNNCell(
        [make_cell() for _ in range(num_layers)], state_is_tuple = True)
    
    outputs = []
    with tf.variable_scope(name):
        initial_state = cell.zero_state(batch_size, tf.float32)
        state = initial_state

        for time_step in range(num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

    return outputs


def conv1d(name, input_dim, output_dim, filter_size, inputs, stride = 1, he_init=True, weightnorm = None, gain = 1., biases = True):
	'''
	inputs: tensor of shape (batch_size, num_channels, width)

	returns: tensor of shape (batch_size, num_channels, width)
	'''

	with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
		
		fan_in = input_dim * filter_size
		fan_out = output_dim * filter_size / stride

		if he_init:
			filters_stdev = tf.math.sqrt(4. / (fan_in + fan_out))
		else:
			filters_stdev = tf.math.sqrt(2. / (fan_in + fan_out))

		filters = tf.get_variable('Filters',
			initializer = tf.random.uniform(
				shape = [filter_size, input_dim, output_dim],
				minval = -filters_stdev * np.sqrt(3),
				maxval = filters_stdev * np.sqrt(3)))

		filters = filters * gain

		if weightnorm == None:
			weightnorm = _default_weightnorm
		if weightnorm:
			norm_values = tf.math.sqrt(tf.math.reduce_sum(
				tf.math.square(filter_values),
				axis =[0,1]))
			target_norms = tf.get_variable('g',initializer = norm_values)
			with tf.variable_scope('weightnorm', reuse = tf.AUTO_REUSE):
				norms = tf.math.sqrt(tf.reduce_sum(
					tf.math.square(filters),
					reduction_indices = [0,1]))
				filters = filters * (target_norms / norms)

		result = tf.nn.conv1d(
			value = inputs, 
			filters = filters,
			stride = stride,
			padding = 'SAME',
			data_format = 'NCW')
		if biases:
			_biases = tf.get_variable(
				'Biases',
				initializer = tf.zeros([output_dim], dtype='float32'))
			result = tf.expand_dims(result, 3)
			result = tf.nn.bias_add(result, _biases, data_format='NCHW')
			result = tf.squeeze(result)

		return result

def linear(
        name, 
        input_dim, 
        output_dim, 
        inputs,
        biases=True,
        initialization=None,
        weightnorm=None,
        gain=1.):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):

        def uniform(stdev, size):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return tf.random.uniform(
            	shape=size,
                minval=-stdev * np.sqrt(3),
                maxval=stdev * np.sqrt(3))

        if initialization == 'lecun':# and input_dim != output_dim):
            # disabling orth. init for now because it's too slow
            weight_values = uniform(
                np.sqrt(1./input_dim),
                (input_dim, output_dim))

        elif initialization == 'glorot' or (initialization == None):

            weight_values = uniform(
                np.sqrt(2./(input_dim+output_dim)),
                (input_dim, output_dim))

        elif initialization == 'he':

            weight_values = uniform(
                np.sqrt(2./input_dim),
                (input_dim, output_dim))

        elif initialization == 'glorot_he':

            weight_values = uniform(
                np.sqrt(4./(input_dim+output_dim)),
                (input_dim, output_dim))

        elif initialization == 'orthogonal' or \
            (initialization == None and input_dim == output_dim):
            
            # From lasagne
            def sample(shape):
                if len(shape) < 2:
                    raise RuntimeError("Only shapes of length 2 or more are "
                                       "supported.")
                flat_shape = (shape[0], np.prod(shape[1:]))
                 # TODO: why normal and not uniform?
                a = tf.random.normal(flat_shape, 0.0, 1.0)
                u, _, v = tf.linalg.svd(a, full_matrices=False)
                # pick the one with the correct shape
                q = u if tf.shape(u) == flat_shape else v
                q = tf.reshape(q, shape)
                return q
            weight_values = sample((input_dim, output_dim))
        
        elif initialization[0] == 'uniform':
        
            weight_values = tf.random.uniform(
            	shape=[input_dim, output_dim],
                minval=-initialization[1],
                maxval=initialization[1])

        else:

            raise Exception('Invalid initialization!')

        weight_values *= gain

        weight = tf.get_variable('W',initializer = weight_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = tf.sqrt(tf.sum(tf.square(weight_values), axis=0))
            # norm_values = np.linalg.norm(weight_values, axis=0)

            target_norms = tf.get_variable('g',initializer = norm_values)

            with tf.variable_scope('weightnorm', reuse = tf.AUTO_REUSE):
                norms = tf.sqrt(tf.reduce_sum(tf.square(weight), reduction_indices=[0]))
                weight = weight * (target_norms / norms)

        # if 'Discriminator' in name:
        #     print "WARNING weight constraint on {}".format(name)
        #     weight = tf.nn.softsign(10.*weight)*.1

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.pack(tf.unpack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            bias = tf.get_variable('b',initializer=tf.zeros(output_dim))
            result = tf.nn.bias_add(result, bias)
        return result