from __future__ import print_function

import numpy as np
from random import shuffle
from six.moves import cPickle as pickle

class BinaryCount:
	pass

def generate():
	'''Generating binary string of length 20'''
	X = ['{0:020b}'.format(i) for i in xrange(2**20)]
	shuffle(X)
	X = np.array([map(int, i) for i in X])
	X = np.reshape(X, [len(X), 20, 1])

	y = []

	for i in X:
		count = np.sum(i[:, 0])
		buffer = np.zeros(21)
		buffer[count] = 1
		y.append(buffer)

	dataset = BinaryCount()
	dataset.X = X
	dataset.y = y
	
	return dataset

def save(dataset, file_name):
	with open('data/' + file_name, 'wb') as f:
		pickle.dump(dataset, f)

def load(file_name):
	with open('data/' + file_name, 'rb') as f:
		dataset = pickle.load(f)

	return dataset

def split(dataset, train_size=10000):
	X = dataset.X
	y = dataset.y

	train_dataset = BinaryCount()
	train_dataset.input = X[:train_size]
	train_dataset.output = y[:train_size]

	test_dataset = BinaryCount()
	test_dataset.input = X[train_size:]
	test_dataset.output = y[train_size:]

	return train_dataset, test_dataset

if __name__ == '__main__':
	dataset = generate()
	train_dataset, test_dataset = split(dataset)

	save(train_dataset, 'train_dataset.pkl')
	save(test_dataset, 'test_dataset.pkl')

	train_dataset = load('train_dataset.pkl')
	train_input = train_dataset.input
	train_output = train_dataset.output

	print(train_input[1], train_output[1])
