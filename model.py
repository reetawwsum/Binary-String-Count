from __future__ import print_function

import tensorflow as tf

from ops import *
from utils import *

class Model:
	'''Two layer LSTM network'''
	def __init__(self, num_units, num_unrollings, batch_size, epochs):
		self.num_units = num_units
		self.num_unrollings = num_unrollings # Sequence length
		self.batch_size = batch_size
		self.epochs = epochs
		self.output_size = num_unrollings + 1

		self.build_model()

	def inference(self):
		# Creating LSTM layer
		cell = tf.nn.rnn_cell.LSTMCell(self.num_units)

		# Creating Unrolled LSTM
		hidden, _ = tf.nn.dynamic_rnn(cell, self.data, dtype=tf.float32)

		# Calculating hidden output at last time step
		hidden = tf.transpose(hidden, [1, 0, 2])
		last = tf.gather(hidden, int(hidden.get_shape()[0]) - 1)

		# Creating variables for output layer
		weight = weight_variable([self.num_units, self.output_size])
		bias = bias_variable([self.output_size])

		# Calculating softmax
		prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

		self.prediction = prediction

	def loss_op(self):
		cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))

		self.loss = cross_entropy

	def train_op(self):
		optimizer = tf.train.AdamOptimizer()

		self.optimizer = optimizer.minimize(self.loss)

	def create_saver(self):
		saver = tf.train.Saver()

		self.saver = saver

	def accuracy(self):
		dataset = Dataset(self.num_unrollings, 'test_dataset')
		test_data = dataset.data
		test_target = dataset.target

		test_predictions = self.predict(test_data)

		correct_prediction = np.equal(test_predictions, np.argmax(test_target, 1))

		return np.mean(correct_prediction)

	def build_model(self):
		self.graph = tf.Graph()

		with self.graph.as_default():
			# Creating placeholder for data and target
			self.data, self.target = placeholder_input(self.num_unrollings, self.output_size)

			# Builds the graph that computes inference
			self.inference()

			# Adding loss op to the graph
			self.loss_op()

			# Adding train op to the graph
			self.train_op()

			# Creating saver
			self.create_saver()

	def train(self):
		with tf.Session(graph=self.graph) as self.sess:
			init = tf.initialize_all_variables()
			self.sess.run(init)
			print('Graph Initialized')

			train_batches = BatchGenerator(self.batch_size, self.num_unrollings)

			for step in xrange(self.epochs * (10000/self.batch_size) + 1):
				train_data, train_target = train_batches.next()
				feed_dict = {self.data: train_data, self.target: train_target}

				_, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

				if not step % 50:
					epoch = step / 10
					self.save(epoch)
					print('Loss at Epoch %d: %f' % (epoch, l))


	def predict(self, test_data):
		with tf.Session(graph=self.graph) as self.sess:
			self.load(3110)
			print('Model Restored')

			predictions = []

			for data in test_data:
				feed_dict = {self.data: np.reshape(data, [1, self.num_unrollings, 1])}

				prediction = self.sess.run(self.prediction, feed_dict=feed_dict)

				predictions.append(np.argmax(prediction))

			return predictions

	def save(self, global_step):
		self.saver.save(self.sess, 'checkpoint/lstm-rnn', global_step=global_step)

	def load(self, global_step):
		self.saver.restore(self.sess, 'checkpoint/lstm-rnn-' + str(global_step))
