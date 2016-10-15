from __future__ import print_function

import tensorflow as tf

from ops import *
from utils import *

class Model:
	'''Two layer LSTM network'''
	def __init__(self, num_units, num_unrollings, batch_size):
		self.num_units = num_units
		self.num_unrollings = num_unrollings # Sequence length
		self.batch_size = batch_size
		self.output_size = num_unrollings + 1

		self.build_model()

	def inference(self, data):
		# Creating LSTM layer
		cell = tf.nn.rnn_cell.LSTMCell(self.num_units)

		# Creating Unrolled LSTM
		hidden, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

		# Calculating hidden output at last time step
		hidden = tf.transpose(hidden, [1, 0, 2])
		last = tf.gather(hidden, int(hidden.get_shape()[0]) - 1)

		# Creating variables for output layer
		weight = weight_variable([self.num_units, self.output_size])
		bias = bias_variable([self.output_size])

		# Calculating softmax
		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

		return prediction

	def loss_op(self, prediction, target):
		cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

		return cross_entropy

	def train_op(self, loss):
		optimizer = tf.train.AdamOptimizer()

		return optimizer.minimize(loss)

	def create_saver(self):
		saver = tf.train.Saver()

		return saver

	def accuracy(self):
		pass

	def build_model(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Creating placeholder for data and target
			data, target = placeholder_input(self.num_unrollings, self.output_size)

			# Builds the graph that computes inference
			prediction = self.inference(data)

			# Adding loss op to the graph
			loss = self.loss_op(prediction, target)

			# Adding train op to the graph
			optimizer = self.train_op(loss)

			# Creating saver
			saver = self.create_saver()

	def train(self):
		with tf.Session(graph=self.graph) as self.sess:
			init = tf.initialize_all_variables()
			self.sess.run(init)
			print('Graph Initialized')

			train_batches = BatchGenerator(self.batch_size, self.num_unrollings)

	def predict(self):
		pass

	def save(self):
		pass

	def load(self):
		pass

if __name__ == '__main__':
	lstm_model = Model(24, 20, 1000)
	lstm_model.train()