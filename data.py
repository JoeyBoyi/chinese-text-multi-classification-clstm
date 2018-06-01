# -*- coding=utf-8 -*-

import xmltodict
import collections
from collections import Counter
import jieba
import jieba.analyse
import jieba.analyse.analyzer
import jieba.posseg as pseg
import re
import numpy as np
from pandas import DataFrame
import pandas as pd
import sys, os, json
from data_deal import data_preprocessing as dp
import data_helper

param_config = "/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/param_config.json"
params = json.loads(open(param_config).read())

class Issue(object):

    def __init__(self, jira_xml):
        with open(jira_xml,'r') as fr:
            self.raw_data = xmltodict.parse(fr.read())
        self._MAX_LEN_ = 100
        self.vocab_size = 20000

    def regulation(self, sentence):
        ''''normalize_digits: Boolean; if true, all digits are replaced by 0s'''
        res = [
        re.compile(r'<[^>]+>',re.S),
        re.compile(r'\&[a-zA-Z]+;',re.S),
        re.compile(r'\n|\t',re.S),
        re.compile(r'\[|\]|\(|\)',re.S),
        re.compile(r'[\s\.\!\/`_,$%^*(+\"\')]+|[+——()?“”！，;；<=>～~{|}。？、~@#￥……&*（）]+|[/.,"=:_-\{\}-、，－、；\\［］;\-（）：。《》〉]+',re.S),
        re.compile(r'【.*】',re.S),    #去除括号和括号内的文字
        re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE),
        re.compile(r'[0-9A-Za-z]+', re.S)   #去除数字和英文
        # re.compile(r'[0-9]+', re.S)   # 仅去除数字
        ]
        for r in res:
            sentence = r.sub('', sentence)
        return sentence

    def parse(self, w2v_embeddings=False):
        items = self.raw_data['list']['Issue']
        
        def content(item):
            d = collections.defaultdict(lambda:'')
            for k in item:
                d[k] = item[k]
            # sentence = str(d['summary']) + str(d['description']) + str(d['comments'])
            sentence = str(d['summary']) + str(d['description'])  
            # 获取文本内容
            sentence = self.regulation(sentence)
            words = dp.Word_segmentation(sentence).clean_words()
            
            length = len(words)
            _lengthes.append(length)
            # if length > self._MAX_LEN_: # 就使用一个默认值，理论上可以使用整体的 95 分位值，先跑一次再说
            #     words = words[0:self._MAX_LEN_]
            issue_type = d['issuetype']
            
            return (issue_type, words)
        
        _lengthes = []
        s = list(map(content, items))
        self._MAX_LEN_ = int(np.percentile(_lengthes, 95))
        
        vc = dp.Vocabulary(s, self.vocab_size)
        x, y, vocabulary, vocab_list, label_list, contents= vc.process_file(max_length=self._MAX_LEN_)
        
        # ID写入文件
        dp.file_processing(params["global_path"]+params["xId_path"]).dataId2file(x)
        dp.file_processing(params["global_path"]+params["yId_path"]).dataId2file(y)
        
        # 是否使用word2vec
        if w2v_embeddings:
            word2vec_path = params["global_path"] + params["word2vec_path"]
            word_embeddings = dp.word2vec().load_embeddings(vocabulary, contents, word2vec_path, size=params["embedding_dim"], min_count=0)
        else:
            word_embeddings = data_helper.load_embeddings(vocabulary, params["embedding_dim"])
        embedding_mat = [word_embeddings[word] for word in vocab_list]
        embedding_mat = np.array(embedding_mat, dtype = np.float32)

        

        print(embedding_mat)
        print(label_list)
        print(len(label_list))
        print("x with shape : {}, preview: \n{}\n".format(x.shape , x[:20]))
        print("y with shape : {}, preview: \n{}\n".format(y.shape , y[:20]))
        return x, y, vocabulary, embedding_mat, label_list

        

if __name__ == '__main__':
    Issue('/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/data/SearchRequest_All.xml').parse(w2v_embeddings=True)

    
