#!/usr/bin/env python
# encoding: utf-8
"""
@author: hwh
@contact: ruc_hwh_2013@163.com
@file: predict.py
@time: 2018/2/7 下午11:27
"""

import os
from gensim import corpora, models
from conf.trainModel_conf import path_dictionary, path_tmp_lsi, path_tmp_lsimodel, path_tmp_predictor
import pickle as pkl
import jieba
from scipy.sparse import csr_matrix

def predict(doc):
    files = os.listdir(path_tmp_lsi)
    dictionary = corpora.Dictionary.load(path_dictionary)
    lsi_file = open(path_tmp_lsimodel, 'rb')
    lsi_model = pkl.load(lsi_file)
    lsi_file.close()
    x = open(path_tmp_predictor, 'rb')
    predictor = pkl.load(x)
    x.close()
    catg_list = []
    for file in files:
        t = file.split('.')[0]
        if t not in catg_list:
            catg_list.append(t)
    # print("原文本内容为：")
    # print(doc)
    demo_doc = list(jieba.cut(doc, cut_all=False))
    demo_bow = dictionary.doc2bow(demo_doc)
    tfidf_model = models.TfidfModel(dictionary=dictionary)
    demo_tfidf = tfidf_model[demo_bow]
    demo_lsi = lsi_model[demo_tfidf]
    data = []
    cols = []
    rows = []
    for item in demo_lsi:
        data.append(item[1])
        cols.append(item[0])
        rows.append(0)
    demo_matrix = csr_matrix((data, (rows, cols))).toarray()
    x = predictor.predict(demo_matrix)
    # print('分类结果为：{x}'.format(x=catg_list[x[0]]))
    return catg_list[x[0]]

if __name__ == '__main__':
    doc = """
    狼牙月，伊人憔悴，我举杯，饮尽了风雪
    """
    print predict(doc)