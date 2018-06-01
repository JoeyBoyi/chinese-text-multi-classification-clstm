# -*- coding=utf-8 -*-

import os
import sys
import json
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
import jieba
import falcon

logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels):
	df = pd.read_csv(test_file, sep='|')
	select = ['Descript']

	df = df.dropna(axis=0, how='any', subset=select)
	test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()

	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	y_ = None
	if 'Category' in df.columns:
		select.append('Category')
		y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()

	not_select = list(set(df.columns) - set(select))
	df = df.drop(not_select, axis=1)
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)
	return x_

trained_dir = "./trained_results_1513061604/"
params, words_index, labels, embedding_mat = load_trained_params(trained_dir)

# graph = tf.Graph()
with tf.Graph().as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)
	sess.as_default()

	cnn_rnn = TextCNNRNN(
					embedding_mat = embedding_mat,
					non_static = params['non_static'],
					hidden_unit = params['hidden_unit'],
					sequence_length = params['sequence_length'],
					max_pool_size = params['max_pool_size'],
					filter_sizes = map(int, params['filter_sizes'].split(",")),
					num_filters = params['num_filters'],
					num_classes = len(labels),
					embedding_size = params['embedding_dim'],
					l2_reg_lambda = params['l2_reg_lambda'])

	checkpoint_file = trained_dir + 'best_model.ckpt'
	saver = tf.train.Saver(tf.all_variables())
	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	saver.restore(sess, checkpoint_file)
	logging.critical('{} has been loaded'.format(checkpoint_file))

print(sess)

def do_predict(x_batch):

	def real_len(batches):
		return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

	def predict_step(x_batch):
		feed_dict = {
			cnn_rnn.input_x: x_batch,
			cnn_rnn.dropout_keep_prob: 1.0,
			cnn_rnn.batch_size: len(x_batch),
			cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
			cnn_rnn.real_len: real_len(x_batch),
		}
		predictions = sess.run([cnn_rnn.predictions], feed_dict)
		return predictions

	batches = data_helper.batch_iter(list(x_batch), params['batch_size'], 1, shuffle=False)

	predictions, predict_labels = [], []
	for x_batch in batches:
		batch_predictions = predict_step(x_batch)[0]
		for batch_prediction in batch_predictions:
			predictions.append(batch_prediction)
			predict_labels.append(labels[batch_prediction])
	print(predict_labels)
	return predict_labels

class Predicts():
	def on_get(self, req, resp):
		words = req.get_param('words') or ' '
		print(words)
		x_ = list(jieba.cut(words))[0:params['sequence_length']]
		x_ = x_ + [' '] * ( params['sequence_length'] - len(x_) )
		x_ = map_word_to_index([ x_ ], words_index)

		x_test = np.asarray(x_)
		result = do_predict(x_test)
		resp.status = falcon.HTTP_200
		resp.body = json.dumps({ "input_sentence": words, "predict_result": result },ensure_ascii=False,indent=2)


predicts = Predicts()
app = falcon.API()
app.add_route('/predicts', predicts)

