#-*-coding:utf-8-*-
# file: main.py
#@author: hwh
#@contact: ruc_hwh_2013@163.com
# Created on 2018/1/21 上午12:41


from __future__ import division
from pyspark import SparkConf, SparkContext
from gensim import corpora, models
from scipy.sparse import csr_matrix
from sklearn import svm
import numpy as np
import os
import re
import jieba
import pickle as pkl
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint


sc = SparkContext('local')
path_doc_root = './data/'  # 根目录 即存放按类分类好的问本纪
path_tmp = './tmp'  # 存放中间结果的位置
path_dictionary = os.path.join(path_tmp, 'THUNews.dict')
path_tmp_tfidf = os.path.join(path_tmp, 'tfidf_corpus')
path_tmp_lsi = os.path.join(path_tmp, 'lsi_corpus')
path_tmp_lsimodel = os.path.join(path_tmp, 'lsi_model.pkl')
path_tmp_predictor = os.path.join(path_tmp, 'predictor.pkl')


class loadFolders(object):   # 迭代器
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):  # if file is a folder
                yield file_abspath


class loadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):
        folders = loadFolders(self.par_path)
        for folder in folders:              # level directory
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb')
                    content = this_file.read().decode('utf8')
                    yield catg, content
                    this_file.close()


def convert_doc_to_wordlist(str_doc, cut_all):
    sent_list = str_doc.split('\n')
    sent_list = map(rm_char, sent_list)  # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all))
                   for part in sent_list]  # 分词
    word_list = sum(word_2dlist, [])
    return word_list


def rm_tokens(words):  # 去掉一些停用次和数字
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:  # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list


def get_stop_words(path='./data/stopWord.txt'):
    file = open(path, 'rb').read().decode('utf8').split('\n')
    return set(file)


def rm_char(text):
    text = re.sub('\u3000', '', text)
    return text


def svm_classify(train_set, train_tag, test_set, test_tag):
    train_data = train_set
    for idx in range(len(train_set)):
        train_data[idx] = np.append(train_set[idx], train_tag[idx])
    train_data = sc.parallelize(train_data)
    test_data = test_set
    for idx in range(len(test_set)):
        test_data[idx] = np.append(test_set[idx], test_tag[idx])
    test_data = sc.parallelize(test_data)
    test_data = test_data.map(
        lambda x: LabeledPoint(x[-1], x[:x.shape[0] - 1]))
    train_data = train_data.map(
        lambda x: LabeledPoint(x[-1], x[:x.shape[0] - 1]))
    model = SVMWithSGD.train(train_data, iterations=150, step=0.8)
    labelsAndPreds = train_data.map(
        lambda p: (
            p.label, model.predict(
                p.features)))
    trainErr = labelsAndPreds.filter(
        lambda lp: lp[0] != lp[1]).count() / float(train_data.count())
    labelsAndPreds_test = test_data.map(
        lambda p: (p.label, model.predict(p.features)))
    testErr = labelsAndPreds_test.filter(
        lambda lp: lp[0] != lp[1]).count() / float(test_data.count())
    #clf = svm.LinearSVC()
    #clf_res = clf.fit(train_set, train_tag)
    #train_pred = clf_res.predict(train_set)
    #test_pred = clf_res.predict(test_set)

    #train_err_num, train_err_ratio = checkPred(train_tag, train_pred)
    #test_err_num, test_err_ratio = checkPred(test_tag, test_pred)

    print('=== 分类训练完毕，分类结果如下 ===')
    print('训练集误差: {e}'.format(e=trainErr))
    print('检验集误差: {e}'.format(e=testErr))

    return model


def checkPred(data_tag, data_pred):
    if data_tag.__len__() != data_pred.__len__():
        raise RuntimeError(
            'The length of data tag and data pred should be the same')
    err_count = 0
    for i in range(data_tag.__len__()):
        if data_tag[i] != data_pred[i]:
            err_count += 1
    err_ratio = err_count / data_tag.__len__()
    return [err_count, err_ratio]


def create_dictionary(path_dictionary, files):
    if not os.path.exists(path_dictionary):
        print('=== 未检测到有词典存在，开始遍历生成词典 ===')
        dictionary = corpora.Dictionary()
        for i, msg in enumerate(files):
            catg = msg[0]
            file = msg[1]
            file = convert_doc_to_wordlist(file, cut_all=False)
            dictionary.add_documents([file])
        # 去掉词典中出现次数过少的
        small_freq_ids = [tokenid for tokenid,
                          docfreq in dictionary.dfs.items() if docfreq < 5]
        dictionary.filter_tokens(small_freq_ids)
        dictionary.compactify()
        dictionary.save(path_dictionary)
    else:
        print('=== 检测到词典已经存在，跳过该阶段 ===')


def train_model():
    dictionary = None
    corpus_tfidf = None
    corpus_lsi = None
    lsi_model = None
    predictor = None
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)
    files = loadFiles(path_doc_root)

    # 生成字典
    create_dictionary(path_dictionary, files)

    # 将文档转化成tfidf
    if not os.path.exists(path_tmp_tfidf):
        print('=== 未检测到有tfidf文件夹存在，开始生成tfidf向量 ===')
        dictionary = corpora.Dictionary.load(path_dictionary)
        os.makedirs(path_tmp_tfidf)
        tfidf_model = models.TfidfModel(dictionary=dictionary)
        corpus_tfidf = {}
        for i, msg in enumerate(files):
            catg = msg[0]
            file = msg[1]
            word_list = convert_doc_to_wordlist(file, cut_all=False)
            file_bow = dictionary.doc2bow(word_list)
            file_tfidf = tfidf_model[file_bow]
            tmp = corpus_tfidf.get(catg, [])
            tmp.append(file_tfidf)
            if tmp.__len__() == 1:
                corpus_tfidf[catg] = tmp
        # 将tfidf中间结果储存起来
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            corpora.MmCorpus.serialize(
                '{f}{s}{c}.mm'.format(
                    f=path_tmp_tfidf,
                    s=os.sep,
                    c=catg),
                corpus_tfidf.get(catg),
                id2word=dictionary)
            print(
                'catg {c} has been transformed into tfidf vector'.format(
                    c=catg))
        print('=== tfidf向量已经生成 ===')
    else:
        print('=== 检测到tfidf向量已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第三阶段，  开始将tfidf转化成lsi
    if not os.path.exists(path_tmp_lsi):
        print('=== 未检测到有lsi文件夹存在，开始生成lsi向量 ===')
        if not dictionary:
            dictionary = corpora.Dictionary.load(path_dictionary)
        if not corpus_tfidf:  # 如果跳过了第二阶段，则从指定位置读取tfidf文档
            print('--- 未检测到tfidf文档，开始从磁盘中读取 ---')
            # 从对应文件夹中读取所有类别
            files = os.listdir(path_tmp_tfidf)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)

            # 从磁盘中读取corpus
            corpus_tfidf = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(
                    f=path_tmp_tfidf, s=os.sep, c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_tfidf[catg] = corpus
            print('--- tfidf文档读取完毕，开始转化成lsi向量 ---')

        # 生成lsi model
        os.makedirs(path_tmp_lsi)
        corpus_tfidf_total = []
        catgs = list(corpus_tfidf.keys())
        for catg in catgs:
            tmp = corpus_tfidf.get(catg)
            corpus_tfidf_total += tmp
        lsi_model = models.LsiModel(
            corpus=corpus_tfidf_total,
            id2word=dictionary,
            num_topics=50)
        # 将lsi模型存储到磁盘上
        lsi_file = open(path_tmp_lsimodel, 'wb')
        pkl.dump(lsi_model, lsi_file)
        lsi_file.close()
        del corpus_tfidf_total  # lsi model已经生成，释放变量空间
        print('--- lsi模型已经生成 ---')

        # 生成corpus of lsi, 并逐步去掉 corpus of tfidf
        corpus_lsi = {}
        for catg in catgs:
            corpu = [lsi_model[doc] for doc in corpus_tfidf.get(catg)]
            corpus_lsi[catg] = corpu
            corpus_tfidf.pop(catg)
            corpora.MmCorpus.serialize(
                '{f}{s}{c}.mm'.format(
                    f=path_tmp_lsi,
                    s=os.sep,
                    c=catg),
                corpu,
                id2word=dictionary)
        print('=== lsi向量已经生成 ===')
    else:
        print('=== 检测到lsi向量已经生成，跳过该阶段 ===')

    # # ===================================================================
    # # # # 第四阶段，  分类
    if not os.path.exists(path_tmp_predictor):
        print('=== 未检测到判断器存在，开始进行分类过程 ===')
        if not corpus_lsi:  # 如果跳过了第三阶段
            print('--- 未检测到lsi文档，开始从磁盘中读取 ---')
            files = os.listdir(path_tmp_lsi)
            catg_list = []
            for file in files:
                t = file.split('.')[0]
                if t not in catg_list:
                    catg_list.append(t)
            # 从磁盘中读取corpus
            corpus_lsi = {}
            for catg in catg_list:
                path = '{f}{s}{c}.mm'.format(f=path_tmp_lsi, s=os.sep, c=catg)
                corpus = corpora.MmCorpus(path)
                corpus_lsi[catg] = corpus
            print('--- lsi文档读取完毕，开始进行分类 ---')

        tag_list = []
        doc_num_list = []
        corpus_lsi_total = []
        catg_list = []
        files = os.listdir(path_tmp_lsi)
        for file in files:
            t = file.split('.')[0]
            if t not in catg_list:
                catg_list.append(t)
        for count, catg in enumerate(catg_list):
            tmp = corpus_lsi[catg]
            tag_list += [count] * tmp.__len__()
            doc_num_list.append(tmp.__len__())
            corpus_lsi_total += tmp
            corpus_lsi.pop(catg)

        # 将gensim中的mm表示转化成numpy矩阵表示
        data = []
        rows = []
        cols = []
        line_count = 0
        for line in corpus_lsi_total:
            for elem in line:
                rows.append(line_count)
                cols.append(elem[0])
                data.append(elem[1])
            line_count += 1
        lsi_matrix = csr_matrix((data, (rows, cols))).toarray()
        # 生成训练集和测试集
        rarray = np.random.random(size=line_count)
        train_set = []
        train_tag = []
        test_set = []
        test_tag = []
        for i in range(line_count):
            if rarray[i] < 0.8:
                train_set.append(lsi_matrix[i, :])
                train_tag.append(tag_list[i])
            else:
                test_set.append(lsi_matrix[i, :])
                test_tag.append(tag_list[i])

        # 生成分类器
        predictor = svm_classify(train_set, train_tag, test_set, test_tag)
        x = open(path_tmp_predictor, 'wb')
        pkl.dump(predictor, x)
        x.close()
    else:
        print('=== 检测到分类器已经生成，跳过该阶段 ===')


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
    print("原文本内容为：")
    print(doc)
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
    x = predictor.predict(sc.parallelize(demo_matrix))
    print('分类结果为：{x}'.format(x=catg_list[x.collect()[0]]))


if __name__ == '__main__':
    doc = '''
    狼牙月，伊人憔悴，我举杯，饮尽了风雪
    '''
    if not os.path.exists(path_tmp):
        train_model()
    predict(doc)
