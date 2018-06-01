# -*- coding=utf-8 -*-

import jieba
import jieba.analyse
import jieba.analyse.analyzer
import jieba.posseg as pseg
import re, json
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from pandas import DataFrame
import pandas as pd
import word2vec
import sys, os, random
from data_deal import tf_idf
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow.contrib.keras as kr
from tensorflow.python.platform import gfile
from imblearn.over_sampling import RandomOverSampler, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsemble

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"

# _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
_START_VOCAB = [_PAD, _UNK]

PAD_ID = 0
# GO_ID = 1
# EOS_ID = 2
UNK_ID = 1

# Regular expressions used to tokenize.
# _DIGIT_RE = re.compile(r"[0-9]+", re.S) # 数字
# _ALPHABET_RE = re.compile(r"[A-Za-z]+", re.S)   # 字母
# _MIX_RE = re.compile(r"[0-9A-Za-z]+", re.S) # 数字字母混合
# _BLANK_RE = re.compile(r"[\s]+", re.S)
# _MIX_CN_RE = re.compile(r'(^[0-9A-Za-z]+[\u4e00-\u9fa5]+$|^[\u4e00-\u9fa5]+[0-9A-Za-z]+$)|([0-9A-Za-z]+[\u4e00-\u9fa5]+[0-9A-Za-z]+)', re.S) # 字符串中有中文、英文、字母混合



param_config = "/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/param_config.json"
params = json.loads(open(param_config).read())

class Word_segmentation(object):
    def __init__(self, sentence):
        '''
        :param sentence: 需要分词的句子
        '''
        self.sentence = sentence

    def get_stopWords(self, stopWords_path):
        with open(stopWords_path, 'r', encoding='utf-8', errors='ignore') as f:
            stopwords = [line.strip() for line in f.readlines()]
        return stopwords

    def cut_words(self, user_dict=None):
        '''
        Jieba分词，并加载自定义词典
        :param user_dict: 自定义词典
        :return:词列表
        '''
        if user_dict:
            jieba.load_userdict(user_dict) #加载分词自定义词典
        cut_t = pseg.cut(self.sentence)
        cut_result = [i.word for i in cut_t]
        return cut_result

    def rm_stopwords(self, cut_result, stopwords):
        '''
        在分词后的结果上去掉停用词
        :param cut_result: 分词后获得的列表
        :param stopwords: 停用词列表
        :return: 去掉停用词后的结果，词列表
        '''
        rm_result = [w for w in cut_result if w not in stopwords]
        return rm_result
    
    '''--------------------------------ignore-------------------------------------'''
    def remove_outliers(self, words):
        # 去空
        while ' ' in words:
            words.remove(' ')
        return words

    def removeTask(self, s):
        '''
        参数s的形式：[(label1, [word1,word2,...]), (label2, [word1,word2,...]), ...]
        '''
        return  [tup for tup in s if not(tup[0] == 'Task')]

    def removelabel_Task(self, alist):
        while 'Task' in alist:
            alist.remove('Task')
        return alist
    '''---------------------------------------------------------------------'''
    
    def clean_words(self):
        stopWords_path = params["global_path"]+params["stopword_path"]
        user_dict = params["global_path"]+params["userdict_path"]
        cut_result = self.cut_words(user_dict=user_dict)  # 分词，自定义词典
        stopwords = self.get_stopWords(stopWords_path)
        words = self.rm_stopwords(cut_result, stopwords)  # 去停留词
        return words

class Vocabulary(object):
    def __init__(self, s, vocab_size):
        self.s = s
        self.vocab_size = vocab_size
        self.vocab_list = []
        self.vocabulary = dict()
        self.label_list = []
        self.label_counter = []
        self.label_vocab = dict()
        self.contents = []
        self.allWords = []
        self.allLabels = []
        self.data2id = []
        self.label2id = []

    def create_vocabulary(self, vocab_path, normalize_digits=True, normalize_alphabet=True, normalize_mix=True, normalize_blank=True):
        """
        建词汇表，存储
        param:allWords: [word1, word2, ...]把每行数据放在一个列表中，有同词
        param:vocab_dir: 词汇表存储目录
        param:vocab_size： 词汇表大小
        """
        vocab = {}
        # if not gfile.Exists(vocab_path):
        for word in self.allWords:
            if word.isdigit():
                word = _DIGIT_RE.sub(r"__数字", word) if normalize_digits else word
            elif word.isalpha():
                word = _ALPHABET_RE.sub(r"__字母", word) if normalize_alphabet else word
            elif word.isalnum():
                word = _MIX_RE.sub(r"__混合", word) if normalize_mix else word
            elif word.isspace():
                word = _BLANK_RE.sub(r"__空格", word) if normalize_blank else word                    
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
        # print(len(vocab))

        self.vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(self.vocab_list) > self.vocab_size:
            self.vocab_list = self.vocab_list[:self.vocab_size]
        file_processing(vocab_path).vocab2file(self.vocab_list)


    def initialize_vocabulary(self, vocab_path):
        """读取词汇表"""
        if gfile.Exists(vocab_path):
            self.vocab_list = file_processing(vocab_path).file2vocab()
            self.vocabulary = dict([(word, index) for (index, word) in enumerate(self.vocab_list)])   #去除字典中的空格
        else:
            raise ValueError("Vocabulary file %s not found.", vocab_path)
    
    def initialize_labels(self):
        self.label_list = [ label[0] for label in Counter(self.allLabels).most_common() ]
        self.label_counter = [label for label in Counter(self.allLabels).most_common()]
        self.label_vocab = dict(zip(self.label_list, range(len(self.label_list))))   # {类别：索引}

    def contents_to_ids(self, with_start=True, with_end=True):
        """
        Args:
            sentence: the sentence in bytes format to convert to token-ids.
            vocabulary: a dictionary mapping tokens to integers.
            tokenizer: a function to use to tokenize each sentence;
                if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.

        Returns:
            a list of integers, the token-ids for the sentence.
        """

        for i, sentence in enumerate(self.contents):  # 文档数
            ids = [self.vocabulary.get(word, UNK_ID) for word in sentence]
            self.label2id.append(self.label_vocab[self.allLabels[i]])
            if with_start:
                ids = [GO_ID] + ids
            if with_end:
                ids =  ids + [EOS_ID]
            self.data2id.append(ids)

    def ids_to_words(self, ids):
        return [self.vocab_list[index] for index in ids if index in range(len(self.vocab_list))]

    def label_ids_to_words(self, ids):
        return [self.label_list[index] for index in ids if index in range(len(self.label_list))]

    def process_file(self, max_length):
        """
        将文件转换为id表示
        param:s:[(label1, [word1,word2,...]), (label2, [word1,word2,...]), ...]
        param:word_to_id: dict, {word:index,...}，不重复的词
        param:cat_to_id: dict， {label:index,...}，不重复的类别
        param:max_length: pad长度
        
        """
        vocab_path = params["global_path"] + params["vocab_path"]
        corpus_path = params["global_path"] + params["corpus_path"]
        labelCounter_path = params["global_path"]+params["labelCounter_path"]
        
        # 语料库写入文件
        file_processing(corpus_path).corpus2file(self.s)
        # 读取语料库, 获取三个变量
        self.contents, self.allWords, self.allLabels = file_processing(corpus_path).file2corpus()
        
        # 创建词汇表
        self.create_vocabulary(vocab_path, normalize_digits=False, normalize_alphabet=False, normalize_mix=False, normalize_blank=False)
        # 初始化词汇表
        self.initialize_vocabulary(vocab_path)
        self.initialize_labels()
        self.contents_to_ids(with_start = False, with_end = False)
        file_processing(labelCounter_path).labelCounter2file(self.label_counter)
        file_processing("/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/dataOut/label2id.txt").labelCounter2file(self.label2id)

        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(self.data2id, max_length, padding='post', truncating='post', value=0)   # post后面补0
        print ("采样前维度：", x_pad.shape)
        # 不平衡数据采样
        print(np.asarray(self.label2id).shape)
        x_res, y_res = ImbalancedSample(x_pad, np.asarray(self.label2id)).randomOverSampling(random_state=42)
        print ("采样后维度：",x_res.shape, y_res.shape)
        y_oh = kr.utils.to_categorical(y_res)  # 将标签转换为one-hot表示
        
        return x_res, y_oh.astype(int), self.vocabulary, self.vocab_list, self.label_list, self.contents + [["_PAD"],["_UNK"]]
    

    def batch_iter(self, x, y, batch_size=64):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def pad_sequences(self, sequences, pad_mark=0):
        """
        :param sequences:
        :param pad_mark:
        :return:
        """
        max_len = max(map(lambda x : len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    # 归一化
    def data_Normalization(self, x):
        """
        :param x: array
        :return: array
        """
        df = DataFrame(x)
        minmax = MinMaxScaler()
        data = minmax.fit_transform(df)
        # std = minmax.fit_transform(df[2])
        # return df[0], ranks, std
        return data

class file_processing(object):
    def __init__(self, fileName):
        self.fileName = fileName
    
    def corpus2file(self, s):
        '''
        参数s的形式：[(label1, [word1,word2,...]), (label2, [word1,word2,...]), ...]
        '''
        with open(self.fileName, encoding='utf-8', mode='w', errors='ignore') as f:
            for tup in s:
                f.write(' '.join(tup[1]) + '\t' + tup[0] + '\n')

    def vocab2file(self, vocab_no_fre):
        '''
        param:vocab_no_fre：[word1, word2, ...], 排序好的词汇表
        '''
        with open(self.fileName, encoding='utf-8', mode='w') as f:
            f.write('\n'.join(vocab_no_fre) + '\n')
    
    def file2corpus(self):
        contents = []; allWords = []; allLabels = []
        with open(self.fileName, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                s = line.strip().split('\t')
                if len(s) == 2: # 没内容的标签删除
                    contents.append(s[0].strip().split())
                    allWords.extend(s[0].strip().split())                
                    allLabels.append(s[1]) # 最后一列是标签

        return contents, allWords, allLabels
    
    def file2vocab(self):
        with open(self.fileName, encoding='utf-8', mode='r') as f:
            words = f.read().strip().split('\n')
        return words

    def dataId2file(self, ids):
        # df = pd.read_csv(input_file, header=None, index_col=None)
        ids_df = DataFrame(ids)
        ids_df.to_csv(self.fileName, header=False, index=False)

    def labelCounter2file(self, labelCounter_list):
        DataFrame(labelCounter_list).to_csv(self.fileName, header=False, index=False)        
    
    def label2id2file(self, label2id):
        with open(self.fileName, mode='w') as f:
            f.write("\n".join(self.label2id))

class TFIDF:
    def __init__(self):
        pass
    
    def fre_item_data(self, alist):
        '''
        把形如[[],[]]这样的变成一个[]
        :param alist:
        :return:
        '''
        final_item = []
        for i in alist:
            for j in i:
                final_item.append(j)
        return final_item

    def count_times(self, rm_result, idf=False):
        '''
        计算每个分词的出现的次数比例
        :param rm_result:分词列表
        :param idf:是计算tf还是idf
        :return:字典，每个词出现的比例
        '''
        times = Counter(rm_result)
        
        word_ratio = {}
        n = float(len(rm_result))
        d = set(rm_result)
        if idf:
            for i in d:
                word_ratio[i] = math.log(n/times[i])
        else:
            for i in d:
                word_ratio[i] = times[i]/n
        return word_ratio

    def tf_idf(self, word_ratio,idf):
        '''
        计算分词对应的tf-idf值
        :param word_ratio: 在本身的比例, {word:ratio,...}
        :param idf: 在整个文档的比例，是个所有的集合,idf值, {word:idf,...}
        :return:返回各个词的tf_idf值, {word:tfidf,...}
        '''
        words_tfidf = {}
        for i in word_ratio:
            if i in idf:
                words_tfidf[i] = word_ratio[i]*idf[i]
            else:
                words_tfidf[i] = 0
        return words_tfidf
    
    def tf_idf_main(self, sentences):
        '''
        计算分词对应的tf-idf值
        :param sentences: 文本数据，分词后词数据：[['a','b','c'],['a','b'],['g'],[''r,'t','r']]
        :return:返回各个词的tf_idf值, {word:tfidf,...}
        '''
        all_item = self.fre_item_data(sentences)
        words_ratio = self.count_times(all_item)
        idf = self.count_times(all_item, idf=True)
        words_tfidf = self.tf_idf(words_ratio,idf)
        return words_tfidf
    
class word2vec:
    def __init__(self):
        pass

    def word2vec_model(self, sentences, output_model_path, size=100, min_count=0):
        '''
        :param content: 文本数据，分词后词数据：[['a','b','c'],['a','b'],['g'],[''r,'t','r']]
        :param output_model_path: word2vec训练模型的输出文件路径
        :param size:词向量长度，default=100
        :param min_count: 低于最小值将舍去，默认为0
        :return:model
        '''
        model = Word2Vec(sentences, size=size,  window=5,  min_count=min_count, workers=6)
        model.save(output_model_path)  
        model = Word2Vec.load(output_model_path)
        return model

    def computer_word2vec_mean(self, line, model):
        sent = {}
        for w in line:
            sent[w] = np.mean(model[w])
        return sent

    def all_w2v_mean_dict(self, sentences, model):
        """
        计算word2vec_mean
        :param sentences: 文本数据，分词后词数据：[['a','b','c'],['a','b'],['g'],[''r,'t','r']]
        :return: [{word1:w2v1,word2:w2v2,....},{word1:w2v1,word2:w2v2,....},...]
        """
        w2v_mean = []
        for line in sentences:
            w2v_mean.append(self.computer_word2vec_mean(line, model))
        return w2v_mean

    def w2v_main(self, sentences, output_model_path, size=100, min_count=0):
        model = self.word2vec_model(sentences, output_model_path, size=100, min_count=0)
        w2v_mean = self.all_w2v_mean_dict(sentences, model)
        return w2v_mean
    
    def load_embeddings(self, vocab, words, output_model_path, size=200, min_count=0):
        """
        计算word_embeddings
        :param vocab: {word:index,...}
        :param words: 文本数据，分词后词数据：[['a','b','c'],['a','b'],['g'],[''r,'t','r']]
        :return: {word1:w2v_array1,word2:w2v_array2,....} 每个词对应的词向量
        """
        model = self.word2vec_model(words, output_model_path, size=size, min_count=min_count)
        word_embeddings = {}
        for w in vocab:
            word_embeddings[w] = model[w]
        return word_embeddings

def w2v_tfidf_weighted(max_length, contents, all_w2v_mean_dict, words_tfidf):
    '''
    :param max_length:矩阵的最大长度
    :param contents: [['a','b','c'],['a','b'],['g'],[''r,'t','r']]    
    :param all_w2v_mean_dict: [{word1:w2v1,word2:w2v2,....},{word1:w2v1,word2:w2v2,....},...]
    :param words_tfidf: {word1:vec2,word2:vec2,...}
    :return:  [[x11,x21,..],
                [x21,x22,...]
                ...
                [xn1,xn2,...]
                ]
    '''
    # 初始化数组
    x = np.zeros((len(contents), max_length))
    for i, line in enumerate(contents):
        for j, word in enumerate(line):
            x[i][j]  = float(all_w2v_mean_dict[i][word] * words_tfidf[word])
    return x


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