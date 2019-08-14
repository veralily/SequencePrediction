'''
Edit time: 2019-01-15
Here is a model to generate event sequence using history event sequence as inputs
A Encoder-Decoder model is established to generate the event suffix(sequence)
Encoder and Decoder type : RNN
Loss Function: the seq2seq loss between predicted event sequence and target event sequence
'''
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

os.environ['CUDA_VISIBLE_DEVICES']='0'

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

class T_Pred(object):
	def __init__(self, config, event_file, time_file, is_training):
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


	def encoder(self, inputs):
		'''
		Encode the inputs and timestamps into hidden representation.
		Using T_GRU cell
		'''
		# with tf.variable_scope("Generator"):
		# 	outputs = utils.build_encoder_graph_gru(
		# 		inputs,
		# 		t,
		# 		self.hidden_size,
		# 		self.num_layers,
		# 		self.batch_size,
		# 		self.num_steps,
		# 		self.keep_prob,
		# 		self.is_training,
		# 		"Encoder.GRU")
		# 	hidden_r = tf.concat(outputs, 1)
		# 	return hidden_r

		'''
		Encode the inputs only into hidden representation.
		Using GRU cell
		'''
		with tf.variable_scope("Generator"):
			outputs = utils.build_encoder_graph_gru(
				inputs,
				self.hidden_size,
				self.num_layers,
				self.batch_size,
				self.num_steps,
				self.keep_prob,
				self.is_training,
				"Encoder.GRU")
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

	def params_with_name(self, name):
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		return [v for v in variables if name in v.name]

	def loss(self, pred_e, real_e):
		logits = []
		labels = []
		for i in range(self.length):
			logits.append(pred_e[:,i,:])
			labels.append(real_e[:,i])

		
		gen_e_cost = loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			logits, 
            labels,
            weights=[tf.ones([self.batch_size]) for i in range(self.length)],
            name="SeqLossByExample")
		gen_e_cost = tf.reduce_mean(gen_e_cost)
		# gen_cost = mae_loss + gen_e_cost
		gen_cost =  gen_e_cost


		gen_params = self.params_with_name('Generator')

		gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5,beta2=0.9).minimize(gen_cost, var_list= gen_params)
		
		return gen_train_op, gen_cost

	def build(self):
		'''
		build the model
		define the loss function
		define the optimization method'''
		self.input_e = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
		self.targets_e = tf.placeholder(tf.int32, [self.batch_size, self.length])
		
		self.inputs_e = tf.nn.embedding_lookup(self.embeddings, self.input_e)

		hidden_r = self.encoder(self.inputs_e)
		self.pred_e = pred_e = self.g_event(hidden_r)

		gen_train_op, gen_cost = self.loss(
			pred_e,
			self.targets_e)

		self.g_train_op = gen_train_op
		self.g_cost = gen_cost

		correct_pred = tf.cast(tf.nn.in_top_k(tf.reshape(pred_e, [self.batch_size*self.length,-1]), tf.reshape(self.targets_e, [-1]), 1), tf.float32)
		self.correct_pred = tf.reduce_sum(correct_pred)
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
			batch_num, e_x_list, e_y_list, _, _ = read_data.data_iterator(
				self.valid_data,
				self.event_to_id,
				self.num_steps,
				self.length,
				self.batch_size)

			iterations = batch_num

			for i in range(iterations):
				
				# d_iters = 5
				if i > 0 and i % (iterations // 10) == 0:
					lr = lr *2./3
				
				feed_dict = {
				self.input_e : e_x_list[i],
				self.targets_e : e_y_list[i]}
				
				_, correct_pred, pred_e, g_cost = sess.run(
					[self.g_train_op, self.correct_pred, self.pred_e,
					self.g_cost],
					feed_dict = feed_dict)

				sum_correct_pred = sum_correct_pred + correct_pred
				sum_iter = sum_iter + 1

				if i % (iterations // 10) == 0:
					print('[epoch: %d, %d] precision: %f, loss: %f' %(
						epoch,
						int(i // (iterations // 10)),
						sum_correct_pred / (sum_iter * self.batch_size * self.length),
						g_cost))
					
			'''
			evaludation
			'''
			batch_num, e_x_list, e_y_list, _, _ = read_data.data_iterator(
				self.valid_data,
				self.event_to_id,
				self.num_steps,
				self.length,
				self.batch_size)

			iterations = batch_num
			sum_correct_pred = 0.0
			sum_iter = 0.0
			
			for i in range(iterations):

				feed_dict = {
				self.input_e : e_x_list[i],
				self.targets_e : e_y_list[i]}

				if i > 0 and i % (iterations // 10) == 0:
					lr = lr *2./3

				correct_pred, pred_e, g_cost = sess.run(
					[self.correct_pred, self.pred_e, self.g_cost],
					feed_dict = feed_dict)
				sum_correct_pred = sum_correct_pred + correct_pred
				sum_iter = sum_iter + 1
				
				if i % (iterations // 10) == 0:		
					print('%f, precision: %f, loss: %f' %(
						i // (iterations // 10),
						sum_correct_pred / (sum_iter * self.batch_size * self.length),
						g_cost))

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
	args = parser.parse_args()

	assert args.logdir[-1] != '/'
	event_file = './T-pred-Dataset/BPIC2017_event.txt'
	time_file = './T-pred-Dataset/BPIC2017_time.txt'
	model_config = get_config(args.mode)
	is_training = args.is_training
	model = T_Pred(model_config, event_file, time_file, is_training)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config = config) as sess:
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		if not args.eval_only:
			model.train(sess, args)
		model.eval(sess, args)

if __name__ == '__main__':
	main()
