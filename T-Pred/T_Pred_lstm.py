from __future__ import print_function
import os
import time
import random
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import utils
import read_data
import model_config

os.environ['CUDA_VISIBLE_DEVICES']='2'

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

class T_Pred(object):
	def __init__(self, config, cell_type, event_file, time_file, is_training):
		self.cell_type = cell_type
		self.event_file = event_file
		self.time_file = time_file
		self.num_layers = config.num_layers
		self.hidden_size = config.hidden_size
		self.g_size = config.g_size
		self.filter_output_dim = config.filter_output_dim
		self.filter_size = config.filter_size
		self.batch_size = config.batch_size
		self.num_steps = config.num_steps
		self.is_training = is_training
		self.keep_prob = config.keep_prob
		self.res_rate = config.res_rate
		self.length = config.output_length
		self.vocab_size = config.vocab_size
		self.learning_rate = config.learning_rate
		self.LAMBDA = config.LAMBDA
		self.delta = config.delta
		self.gamma = config.gamma
		self.event_to_id = read_data.build_vocab(self.event_file)
		self.train_data, self.valid_data, self.test_data = read_data.data_split(
			event_file, time_file)
		self.embeddings = tf.get_variable(
          "embedding", [self.vocab_size, self.hidden_size], dtype=tf.float32)
		self.build()


	def encoder(self, cell_type, inputs, t):
		'''
		Encode the inputs and timestamps into hidden representation.
		Using T_GRU cell
		'''
		with tf.variable_scope("Generator"):
			outputs = utils.build_encoder_graph_t(
				cell_type,
				inputs,
				t,
				self.hidden_size,
				self.num_layers,
				self.batch_size,
				self.num_steps,
				self.keep_prob,
				self.is_training,
				"Encoder" + cell_type)
			hidden_r = tf.concat(outputs, 1)
			return hidden_r


	def g_event(self, hidden_r):
		'''
		The generator model for time and event
		'''
		with tf.variable_scope("Generator_E"):
			outputs = utils.build_rnn_graph(
				hidden_r,
				self.num_layers,
				self.g_size,
				self.batch_size,
				self.length,
				"G_E.RNN")
			output = tf.reshape(tf.concat(outputs, 1), [-1, self.g_size])
			output = utils.linear('G_E.Output', self.g_size, self.vocab_size, output)
			logits = tf.reshape(output, [self.batch_size, self.length, self.vocab_size])
			return logits

	def g_time(self, hidden_r):
		'''
		The generator model for time and event
		'''
		with tf.variable_scope('Generator_T'):
			outputs = utils.build_rnn_graph(
				hidden_r,
				self.num_layers,
				self.hidden_size,
				self.batch_size,
				self.length,
				"G_T.RNN")
			output = tf.reshape(tf.concat(outputs, 1), [-1, self.g_size])
			output = utils.linear('G_T.Output', self.g_size, 1, output)
			logits = tf.reshape(output, [self.batch_size, self.length, 1])
			return logits

	def discriminator(self, inputs_logits, num_blocks=3, use_bias=False, num_classes=1):
		'''
		The discriminator to score the distribution of time and event
		If the time is consistent with the history times, give high score.
		If it is on the constant, give low score.

		Implementation:
		CNN

		'''
		with tf.variable_scope('Discriminator'):
			inputs = tf.transpose(inputs_logits, [0,2,1])
			output = utils.conv1d('D.Input',1, self.filter_output_dim, self.filter_size, inputs)
			output = self.res_block('D.1', output)
			output = self.res_block('D.2', output)
			output = self.res_block('D.3', output)
			output = self.res_block('D.4', output)
			output = self.res_block('D.5', output)
			output = tf.reshape(output, [-1, self.num_steps * self.filter_output_dim])
			output = utils.linear('D.Output',self.num_steps * self.filter_output_dim, 1, output)
			return output

	def res_block(self, name, inputs):
		output = inputs
		output = tf.nn.relu(output)
		output = utils.conv1d(name+'.1', self.filter_output_dim, self.filter_output_dim, 5, output)
		output = tf.nn.relu(output)
		output = utils.conv1d(name+'.2', self.filter_output_dim, self.filter_output_dim, 5, output)
		return inputs + (self.res_rate * output)

	def params_with_name(self, name):
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		return [v for v in variables if name in v.name]

	def loss_with_wasserstein(self, pred_e, pred_t, real_e, real_t, input_t, sample_t, sample_target_t):
		logits = []
		labels = []
		for i in range(self.length):
			logits.append(pred_e[:,i,:])
			labels.append(real_e[:,i])

		if self.cell_type == 'T_LSTMCell':
			variable_content_e = self.params_with_name('time_gate_t1')
		else:
			variable_content_e = None

		sample_t_concat = tf.expand_dims(tf.concat([sample_t, sample_target_t],1),2)
		disc_real = self.discriminator(sample_t_concat)
		pred_t_concat = tf.concat([tf.expand_dims(input_t, 2), pred_t],1)
		disc_fake = self.discriminator(pred_t_concat)

		disc_cost_1 = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
		# WGAN lipschitz-penalty
		alpha = tf.random_uniform(
			shape = [self.batch_size,1,1],
			minval = 0.0,
			maxval = 1.0)

		differences = pred_t_concat - sample_t_concat
		interpolates = sample_t_concat + (alpha * differences)
		gradients = tf.gradients(self.discriminator(interpolates), [interpolates])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
		gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
		disc_cost = disc_cost_1 + self.LAMBDA * gradient_penalty

		# mae_loss = tf.losses.mean_squared_error(real_t, pred_t)
		# gen_t_cost = -tf.reduce_mean(disc_fake) + self.gamma * mae_loss
		gen_t_cost = -tf.reduce_mean(disc_fake)
		
		gen_e_cost = loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			logits, 
            labels,
            weights=[tf.ones([self.batch_size]) for i in range(self.length)],
            name="SeqLossByExample")
		gen_e_cost = tf.reduce_mean(gen_e_cost)
		gen_cost = gen_t_cost + self.delta * gen_e_cost


		gen_params = self.params_with_name('Generator')
		disc_params = self.params_with_name('Discriminator')

		gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9).minimize(gen_cost, var_list= gen_params)
		disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9).minimize(disc_cost, var_list = disc_params)
		
		# constraint the weight of t for t1 to be negative
		if variable_content_e is not None:
			clip_op = []
			for v in variable_content_e:
				clip_op.append(tf.assign(v, tf.clip_by_value(v, -np.infty, 0)))
		else:
			clip_op = None

		return gen_train_op, disc_train_op, clip_op, gen_cost, disc_cost, gen_t_cost, gen_e_cost, disc_cost_1, gradient_penalty

	def build(self):
		'''
		build the model
		define the loss function
		define the optimization method'''
		self.input_e =  tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
		self.targets_e = tf.placeholder(tf.int32, [self.batch_size, self.length])
		self.inputs_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
		self.target_t = tf.placeholder(tf.float32, [self.batch_size, self.length])
		self.targets_t = tf.expand_dims(self.target_t, 2)
		self.sample_t = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
		self.sample_target_t = tf.placeholder(tf.float32, [self.batch_size, self.length])
		
		self.inputs_e = tf.nn.embedding_lookup(self.embeddings, self.input_e)

		hidden_r = self.encoder(self.cell_type, self.inputs_e, self.inputs_t)
		self.pred_e = pred_e = self.g_event(hidden_r)
		self.pred_t = pred_t = self.g_time(hidden_r)

		gen_train_op, disc_train_op, clip_op, gen_cost, disc_cost, gen_t_cost, gen_e_cost, disc_cost_1, gradient_penalty = self.loss_with_wasserstein(
			pred_e,
			pred_t,
			self.targets_e,
			self.targets_t,
			self.inputs_t,
			self.sample_t,
			self.sample_target_t)

		self.g_train_op = gen_train_op
		self.d_train_op = disc_train_op
		self.clip_op = clip_op
		self.g_cost = gen_cost
		self.d_cost = disc_cost
		self.gen_e_cost = gen_e_cost
		self.gen_t_cost = gen_t_cost
		self.disc_cost_1 = disc_cost_1
		self.gradient_penalty = gradient_penalty

		print('pred_e shape: ')
		print(pred_e.get_shape())
		print('targets_e shape: ')
		print(self.targets_e.get_shape())

		correct_pred = tf.cast(tf.nn.in_top_k(tf.reshape(pred_e, [self.batch_size*self.length,-1]), tf.reshape(self.targets_e, [-1]), 1), tf.float32)
		self.correct_pred = tf.reduce_sum(correct_pred)
		self.deviation = tf.reduce_mean(tf.abs(tf.squeeze(pred_t - self.targets_t)))
		self.saver = tf.train.Saver(max_to_keep=None)

	def train(self, sess, args):
		self.logdir = args.logdir + parse_time()
		while os.path.exists(self.logdir):
			time.sleep(random.randint(1,5))
			self.logdir = args.logdir + parse_time()
		os.makedirs(self.logdir)

		if not os.path.exists('%s/logs' % self.logdir):
			os.makedirs('%s/logs' % self.logdir)

		if args.weights is not None:
			self.saver.restore(sess, args.weights)

		lr = self.learning_rate

		for epoch in range(args.iters):
			'''training'''
			sum_correct_pred = 0.0
			sum_iter = 0.0
			sum_deviation = 0.0

			batch_num, e_x_list, e_y_list, t_x_list, t_y_list = read_data.data_iterator(
				self.train_data,
				self.event_to_id,
				self.num_steps,
				self.length,
				self.batch_size)
			
			iterations = batch_num // 6
			print(iterations)

			
			sample_t_x_list = list(t_x_list)
			sample_t_y_list = list(t_y_list)
			np.random.shuffle(sample_t_x_list)
			np.random.shuffle(sample_t_y_list)
			print(len(t_x_list))
			print(len(sample_t_x_list))

			for i in range(iterations):
				
				d_iters = 5
				if i > 0 and i % (iterations // 10) == 0:
					lr = lr *2./3
				
				for j in range(d_iters):
					feed_dict = {
					self.input_e : e_x_list[i*6+j],
					self.inputs_t : t_x_list[i*6+j],
					self.target_t : t_y_list[i*6+j],
					self.targets_e : e_y_list[i*6+j],
					self.sample_t : sample_t_x_list[i*6+j],
					self.sample_target_t : sample_t_y_list[i*6+j]}

					_ = sess.run(self.d_train_op, feed_dict = feed_dict)
				
				feed_dict = {
				self.input_e : e_x_list[i*6+d_iters],
				self.inputs_t : t_x_list[i*6+d_iters],
				self.target_t : t_y_list[i*6+d_iters],
				self.targets_e : e_y_list[i*6+d_iters],
				self.sample_t : sample_t_x_list[i*6+d_iters],
				self.sample_target_t : sample_t_y_list[i*6+d_iters]}
				
				_, correct_pred, deviation, pred_e, pred_t, d_loss, g_loss, gen_e_cost, gen_t_cost, disc_cost_1, gradient_penalty = sess.run(
					[self.g_train_op,self.correct_pred, self.deviation, self.pred_e, self.pred_t, self.d_cost, self.g_cost,
					self.gen_e_cost, self.gen_t_cost,self.disc_cost_1, self.gradient_penalty],
					feed_dict = feed_dict)

				sess.run(self.clip_op)

				sum_correct_pred = sum_correct_pred + correct_pred
				sum_iter = sum_iter + self.length
				sum_deviation = sum_deviation + deviation

				if i % (iterations // 10) == 0:
					print('[epoch: %d, %d] precision: %f, deviation: %f, d_loss: %f, g_loss: %f' %(
						epoch,
						int(i // (iterations // 10)),
						sum_correct_pred / (sum_iter * self.batch_size),
						sum_deviation / sum_iter,
						d_loss,
						g_loss))
					print()
					print('gen_e_loss: %f, gen_t_loss: %f, d_1_loss: %f, g_penal_loss: %f' %(
						gen_e_cost,
						gen_t_cost,
						disc_cost_1,
						gradient_penalty))
					print()
	
				
			'''
			evaludation
			'''
			
			batch_num, e_x_list, e_y_list, t_x_list, t_y_list = read_data.data_iterator(
				self.valid_data,
				self.event_to_id,
				self.num_steps,
				self.length,
				self.batch_size)

			sample_t_x_list = list(t_x_list)
			sample_t_y_list = list(t_y_list)
			np.random.shuffle(sample_t_x_list)
			np.random.shuffle(sample_t_y_list)
			
			iterations = batch_num
			sum_correct_pred = 0.0
			sum_iter = 0.0
			sum_deviation = 0.0
			
			for i in range(iterations):
				feed_dict = {
				self.input_e : e_x_list[i],
				self.inputs_t : t_x_list[i],
				self.target_t : t_y_list[i],
				self.targets_e : e_y_list[i],
				self.sample_t : sample_t_x_list[i],
				self.sample_target_t : sample_t_y_list[i]}

				if i > 0 and i % (iterations // 10) == 0:
					lr = lr *2./3

				correct_pred, deviation, pred_e, pred_t, d_loss, g_loss, gen_e_cost, gen_t_cost, disc_cost_1, gradient_penalty = sess.run(
					[self.correct_pred, self.deviation, self.pred_e, self.pred_t, self.d_cost, self.g_cost,
					self.gen_e_cost, self.gen_t_cost,self.disc_cost_1, self.gradient_penalty],
					feed_dict = feed_dict)
				sum_correct_pred = sum_correct_pred + correct_pred
				sum_iter = sum_iter + self.length
				sum_deviation = sum_deviation + deviation

				if i % (iterations // 10) == 0:
					print('%f, precision: %f, deviation: %f, d_loss: %f, g_loss: %f' %(
						i // (iterations // 10),
						sum_correct_pred / (sum_iter * self.batch_size),
						sum_deviation / sum_iter,
						d_loss,
						g_loss))

		self.save_model(sess, self.logdir, args.iters)

	def save_model(self, sess, logdir, counter):
		ckpt_file = '%s/model-%d.ckpt' % (logdir, counter)
		print('Checkpoint: %s' % ckpt_file)
		self.saver.save(sess, ckpt_file)

def get_config(config_mode):
	"""Get model config."""
	config = None
	if config_mode == "small":
		config = model_config.SmallConfig()
	elif config_mode == "medium":
		config = model_config.MediumConfig()
	elif config_model == "large":
		config = model_config.LargeConfig()
	elif config_mode == "test":
		config = model_config.TestConfig()
	else:
		raise ValueError("Invalid model: %s", config_mode)
	return config

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='small', type=str)
	parser.add_argument('--is_training', default=True, type=bool)
	parser.add_argument('--weights', default=None, type=str)
	parser.add_argument('--gpu', default=0, type=int)
	parser.add_argument('--eval_only', default=False, action='store_true')
	parser.add_argument('--logdir', default='log/log_kick', type=str)
	parser.add_argument('--iters', default=15, type=int)
	parser.add_argument('--cell_type', default='T_LSTMCell', type=str)
	args = parser.parse_args()

	assert args.logdir[-1] != '/'
	event_file = './T-pred-Dataset/IPTV_event.txt'
	time_file = './T-pred-Dataset/IPTV_time.txt'
	model_config = get_config(args.mode)
	is_training = args.is_training
	cell_type = args.cell_type
	model = T_Pred(model_config, cell_type, event_file, time_file, is_training)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		if not args.eval_only:
			model.train(sess, args)
		model.eval(sess, args)

if __name__ == '__main__':
	main()
