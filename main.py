from __future__ import print_function

from model import *

flags = tf.app.flags
flags.DEFINE_integer('num_units', 24, 'Number of units in LSTM layer')
flags.DEFINE_integer('num_unrollings', 20, 'Input sequence length')
flags.DEFINE_integer('batch_size', 1000, 'The size of training batch')
flags.DEFINE_integer('epochs', 3130, 'Epochs to train')
flags.DEFINE_boolean('train', False, 'True for training, False for testing')
flags.DEFINE_string('dataset_dir', 'data', 'Directory name to save the dataset')
flags.DEFINE_string('checkpoint_dir', 'checkpoint', 'Directory name to save the checkpoint')
flags.DEFINE_string('batch_dataset_type', 'train_dataset', 'Dataset used for generating training batches')
flags.DEFINE_string('accuracy_dataset_type', 'test_dataset', 'Dataset used for generating accuracy')
flags.DEFINE_string('model_name', 'lstm-rnn', 'Name of the model')
flags.DEFINE_integer('restore_model', 3110, 'Model to restore to calculate accuracy')
FLAGS = flags.FLAGS

def main(_):
	model = Model(FLAGS)

	if FLAGS.train:
		model.train()
	else:
		print(model.accuracy())

if __name__ == '__main__':
	tf.app.run()
