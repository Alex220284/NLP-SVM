#-*-coding:utf-8-*-
#file: split.py
#@author: hwh
#@contact: ruc_hwh_2013@163.com
#Created on 2018/1/21 上午12:34

'''
将两类歌曲歌词分别放入文件夹，并且一首歌一个txt文件
'''

import pandas as pd
import numpy as np

liuxing = pd.read_excel('./data/liuxing.xls')
gufeng = pd.read_excel('./data/gufeng.xls')

for idx in range(liuxing.shape[0]):
    if (len(liuxing.iloc[idx,1])>15):
        with open('./data/流行/'+str(idx)+'.txt','w') as f:
            f.write(liuxing.iloc[idx,1].encode('utf8') )

for idx in range(gufeng.shape[0]):
    if (len(gufeng.iloc[idx, 1]) > 15):
        with open('./data/古风/' + str(idx) + '.txt', 'w') as f:
            f.write(gufeng.iloc[idx, 1].encode('utf8') )
