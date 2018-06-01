# -*- coding=utf-8 -*-

import re
import os
import sys
import time
import xmltodict
import collections
import pandas as pd
import numpy as np
from collections import Counter
from pandas import DataFrame
from tensorflow.contrib import learn
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsemble
class Issue(object):

    def __init__(self, jira_xml):
        with open(jira_xml,'r') as fr:
            self.raw_data = xmltodict.parse(fr.read())
        self.sentences = []
        self.all_labels = []
        self.labels_id = []
        self.label_dict = dict()
    def regulation(self, sentence):
        ''''normalize_digits: Boolean; if true, all digits are replaced by 0s'''
        res = [
        re.compile(r'<[^>]+>',re.S),
        re.compile(r'\&[a-zA-Z]+;',re.S),
        re.compile(r'\n|\t',re.S),
        re.compile(r'\[|\]|\(|\)',re.S),
        re.compile(r'[\.\!\/`_,$%^*(+\"\')]+|[+——()?“”！，;；<=>～~{|}。？、~@#￥……&*（）]+|[/.,"=:_-\{\}-、，－、；\\［］;\-（）：。《》〉]+',re.S),
        re.compile(r'【.*】',re.S),    #去除括号和括号内的文字
        re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        # re.compile(r'[0-9A-Za-z]+', re.S)   #去除数字和英文
        # re.compile(r'[0-9]+', re.S)   # 仅去除数字
        ]
        for r in res:
            sentence = r.sub('', sentence)
        return sentence

    def load_data(self, sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True):
        items = self.raw_data['list']['Issue']
        print('Building dataset ...')
        start = time.time()
        
        def content(item):
            d = collections.defaultdict(lambda:'')
            for k in item:
                d[k] = item[k]
            # sentence = str(d['summary']) + str(d['description']) + str(d['comments'])
            sent = str(d['summary']) + str(d['description'])  
            # 获取文本内容
            sent = self.regulation(sent).strip()
            
            # 去停用词
            if sw_path is not None:
                sw = self._stop_words(sw_path)
            else:
                sw = None
        
            if language == 'ch':
                sent = self._tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
            elif language == 'en':
                sent = sent.lower()
            else:
                raise ValueError('language should be one of [ch, en].')

            sent = self._clean_data(sent, sw, language=language)  # Remove stop words and special characters
            if len(sent) < 1:
                return

            if language == 'ch':
                sent = self._word_segmentation(sent)
            
            if sent.isspace():              # 去空串
                return
            
            self.sentences.append(sent.strip())
            issue_type = d['issuetype']
            self.all_labels += [ issue_type ]

            return (issue_type, sent)

        s = list(map(content, items))
        self.labels = [ label[0] for label in Counter(self.all_labels).most_common() ]
        self.label_dict = dict(zip(self.labels, range(len(self.labels))))
        # print(self.sentences)
        for label in self.all_labels:
            self.labels_id.append(self.label_dict[label])
        self.labels_id = np.array(self.labels_id)
        print(len(set(self.labels_id)))
        print(self.labels_id.shape)
        # Real lengths
        lengths = np.asarray(list(map(len, [sent.strip().split(' ') for sent in self.sentences])))
        
        if max_length == 0:
            max_length = int(np.percentile(lengths, 95))

        # Extract vocabulary from sentences and map words to indices
        print(len(self.sentences))
        print(max_length)
        if vocab_processor is None:
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
            data = np.asarray(list(vocab_processor.fit_transform(self.sentences)))
        else:
            data = np.asarray(list(vocab_processor.transform(self.sentences)))

        data_size_before = len(data)

        # 采样前：data, self.labels_id, 采样前：data, self.labels_id
        print ("采样前维度：", data.shape)
        data, self.labels_id = ImbalancedSample(data, np.asarray(self.labels_id)).randomOverSampling(random_state=42)
        print ("采样后维度：", data.shape, self.labels_id.shape)

        data_size = len(data)
        lengths = np.asarray(lengths.tolist() + [max_length]*(data_size-data_size_before))


        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
            self.labels_id = self.labels_id[shuffle_indices]
            lengths = lengths[shuffle_indices]

        end = time.time()
        # 写入语料和类别
        file_processing('/home/xw/codeRepository/NLPspace/QA/jira_issue_clstm_multi_clf/dataOut/sentences.txt').sentences2file(self.sentences, self.all_labels)
        # 写入词汇表        
        file_processing('/home/xw/codeRepository/NLPspace/QA/jira_issue_clstm_multi_clf/dataOut/vocab.txt').vocab2file(vocab_processor.vocabulary_._mapping)
        
        print('Dataset has been built successfully.')
        print('Run time: {}'.format(end - start))
        print('Number of sentences: {}'.format(len(data)))
        print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
        print('Max document length: {}\n'.format(vocab_processor.max_document_length))

        return data, self.labels_id, lengths, vocab_processor, self.label_dict

    # --------------- Private Methods ---------------

    def _tradition_2_simple(self, sent):
        """ Convert Traditional Chinese to Simplified Chinese """
        # Please download langconv.py and zh_wiki.py first
        # langconv.py and zh_wiki.py are used for converting between languages
        try:
            import langconv
        except ImportError as e:
            error = "Please download langconv.py and zh_wiki.py at "
            error += "https://github.com/skydark/nstools/tree/master/zhtools."
            print(str(e) + ': ' + error)
            sys.exit()

        return langconv.Converter('zh-hans').convert(sent)


    def _word_segmentation(self, sent):
        """ Tokenizer for Chinese """
        import jieba
        sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
        return re.sub(r'\s+', ' ', sent)


    def _stop_words(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            sw = list()
            for line in f:
                sw.append(line.strip())

        return set(sw)


    def _clean_data(self, sent, sw, language='ch'):
        """ Remove special characters and stop words """
        if language == 'ch':
            # sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。]", " ", sent)
            sent = re.sub(r"[^\u4e00-\u9fa5！？，。]", " ", sent)        
            sent = re.sub('！{2,}', '！', sent)
            sent = re.sub('？{2,}', '！', sent)
            sent = re.sub('。{2,}', '。', sent)
            sent = re.sub('，{2,}', '，', sent)
            sent = re.sub('\s{2,}', ' ', sent)
        if language == 'en':
            sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
            sent = re.sub(r"\'s", " \'s", sent)
            sent = re.sub(r"\'ve", " \'ve", sent)
            sent = re.sub(r"n\'t", " n\'t", sent)
            sent = re.sub(r"\'re", " \'re", sent)
            sent = re.sub(r"\'d", " \'d", sent)
            sent = re.sub(r"\'ll", " \'ll", sent)
            sent = re.sub(r",", " , ", sent)
            sent = re.sub(r"!", " ! ", sent)
            sent = re.sub(r"\(", " \( ", sent)
            sent = re.sub(r"\)", " \) ", sent)
            sent = re.sub(r"\?", " \? ", sent)
            sent = re.sub(r"\s{2,}", " ", sent)
        if sw is not None:
            sent = "".join([word for word in sent if word not in sw])

        return sent

def batch_iter(data, labels, lengths, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """
    assert len(data) == len(labels) == len(lengths)

    data_size = len(data)
    epoch_length = data_size // batch_size

    for _ in range(num_epochs):
        for i in range(epoch_length):
            start_index = i * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = labels[start_index: end_index]
            sequence_length = lengths[start_index: end_index]

            yield xdata, ydata, sequence_length

class file_processing(object):
    def __init__(self, fileName):
        self.fileName = fileName
    
    def sentences2file(self, sentences, labels):
        '''
        param:sentences：['word1 word2 ...', 'word1 word2 ...', ...]
        '''
        with open(self.fileName, encoding='utf-8', mode='w', errors='ignore') as f:
            for i, sent in enumerate(sentences):
                f.write(sent + '\t' + labels[i] + '\n')

    def vocab2file(self, vocabulary):
        '''
        param:vocabulary:{word1:index1, word2:index2, ...}
        '''
        with open(self.fileName, encoding='utf-8', mode='w') as f:
            f.write('\n'.join(list(vocabulary.keys())) + '\n')

class ImbalancedSample(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def randomOverSampling(self, random_state=42):
        # ros = RandomOverSampler(random_state = 3, ratio = {1:5229, 0:52290}) # 按比例抽取样本
        ros = RandomOverSampler(ratio='minority',random_state=random_state)
        x_res, y_res = ros.fit_sample(self.x, self.y)
        return x_res, y_res
    
    def adasyn(self, random_state=42):
        ada = ADASYN(ratio='minority',random_state=random_state)
        x_res, y_res = ada.fit_sample(self.x, self.y)
        return x_res, y_res

    def smoteEnn(self, random_state=42):
        sme = SMOTEENN(random_state=random_state)
        x_res, y_res = sme.fit_sample(self.x, self.y)
        return x_res, y_res

    def smoteTomek(self, random_state=42):
        smt = SMOTETomek(random_state=random_state)
        x_res, y_res = smt.fit_sample(self.x, self.y)
        return x_res, y_res
    
    def easyEnsemble(self, random_state=42):
        ee = EasyEnsemble(random_state=random_state)
        x_res, y_res = ee.fit_sample(self.x, self.y)
        return x_res, y_res



if __name__ == '__main__':
    Issue('/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/data/SearchRequest_All.xml').load_data(sw_path=None, min_frequency=0, max_length=0, language='ch', vocab_processor=None, shuffle=True)

    
