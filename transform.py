#-*-coding:utf-8-*-
# file: transform.py
#@author: hwh
#@contact: ruc_hwh_2013@163.com
# Created on 2018/1/20 下午6:42

'''
处理读取的两个excel数据，转化为txt文件，并除去没有歌词的样本
'''

import pandas as pd
import numpy as np


count_liuxing = 0
count_gufeng = 0
train_count = 0
test_count = 0

train = open('./data/train.txt', 'w')
test = open('./data/test.txt', 'w')
np.random.seed(1)

liuxing = pd.read_excel('./data/liuxing.xls')
split_liuxing = np.random.choice(
    np.array([True, False]), liuxing.shape[0], p=[0.8, 0.2])
gufeng = pd.read_excel('./data/gufeng.xls')
split_gufeng = np.random.choice(
    np.array([True, False]), gufeng.shape[0], p=[0.8, 0.2])

for idx in range(liuxing.shape[0]):
    if len(liuxing.iloc[idx, 1]) > 15:
        if split_liuxing[idx]:
            train.write(liuxing.iloc[idx, 1].encode(
                'utf8') + '\t' + liuxing.iloc[idx, 2].encode('utf8') + '\n')
            train_count += 1
        else:
            test.write(liuxing.iloc[idx, 1].encode(
                'utf8') + '\t' + liuxing.iloc[idx, 2].encode('utf8') + '\n')
            test_count += 1
        count_liuxing += 1
for idx in range(gufeng.shape[0]):
    if len(gufeng.iloc[idx, 1]) > 15:
        if split_gufeng[idx]:
            train.write(gufeng.iloc[idx, 1].encode(
                'utf8') + '\t' + gufeng.iloc[idx, 2].encode('utf8') + '\n')
            train_count += 1
        else:
            test.write(gufeng.iloc[idx, 1].encode(
                'utf8') + '\t' + gufeng.iloc[idx, 2].encode('utf8') + '\n')
            test_count += 1
        count_gufeng += 1
train.close()
test.close()

print '流行音乐数量：%d' % count_liuxing
print '古风音乐数量：%d' % count_gufeng
print '总数量：%d' % (count_liuxing + count_gufeng)
print '训练集数量：%d' % train_count
print '测试集数量：%d' % test_count


