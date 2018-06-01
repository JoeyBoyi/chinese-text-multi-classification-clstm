# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pickle as pkl
import tensorflow as tf
import data_helper
from tensorflow.contrib import learn
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf
from clstm_classifier import clstm_clf
import jieba
import falcon


# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.flags.DEFINE_string('file_path', None, 'File path')
tf.flags.DEFINE_string('model_file', None, 'Model file')

FLAGS = tf.flags.FLAGS

# Restore parameters
# ============================================================

# load vocabulary
vocab_path = os.path.join(FLAGS.file_path, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# {词：索引}
w_2_idx = vocab_processor.vocabulary_._mapping
# {索引：词}
idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))

labels = json.loads(open(FLAGS.file_path + 'labels.json').read())

# load hyperparameters
params_file = open(os.path.join(FLAGS.file_path, 'params.pkl'), 'rb')
params = pkl.load(params_file)
params_file.close()
params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']

# initialize config with loaded hyperparameters
class config():
    max_length = params['max_length']
    num_classes = params['num_classes']
    vocab_size = params['vocab_size']
    embedding_size = params['embedding_size']
    filter_sizes = params['filter_sizes']
    num_filters = params['num_filters']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    l2_reg_lambda = params['l2_reg_lambda']
    batch_size = params['batch_size']
    keep_prob = 1.0

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        clstm = clstm_clf(config)
        # sess.run(tf.global_variables_initializer())
        # Restore model
        checkpoint_file = os.path.join(FLAGS.file_path, 'model/', FLAGS.model_file)
        saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) 
        def do_predict(x_batch, real_len):
            def predict_step(x_batch, real_len):
                feed_dict = {
                    clstm.input_x: x_batch,
                    clstm.keep_prob: config.keep_prob,
                    clstm.batch_size: len(x_batch),
                    clstm.sequence_length: real_len
                }
                predictions = sess.run([clstm.predictions], feed_dict)
                return predictions

            batches = batch_iter(list(x_batch), params['batch_size'], 1, shuffle=False)

            predictions, predict_labels = [], []
            for x_batch in batches:
                batch_predictions = predict_step(x_batch, real_len)[0]
                for batch_prediction in batch_predictions:
                    predictions.append(batch_prediction)
                    predict_labels.append(labels[batch_prediction])
            return predict_labels

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

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


def main():
    while(1):
        print('Please input your sentence:')
        words = input()
        if words == '' or words.isspace():
            print('See you next time!')
            break
        else:
            words = words.strip()
            x_ = list(jieba.cut(words))[0:params['max_length']]
            real_len = [len(x_)]
            x_ = x_ + ['<UNK>'] * ( params['max_length'] - len(x_) )
            x_ = map_word_to_index([ x_ ], w_2_idx)
            x_test = np.asarray(x_)
            result = do_predict(x_test, real_len)
            
            print("input_sentences is:\n {}, \nand the predict_result is:\n {}.".format(words, result))


if __name__ == '__main__':
    main()
