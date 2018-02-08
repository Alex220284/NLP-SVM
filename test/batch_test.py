#!/usr/bin/env python
# encoding: utf-8
"""
@author: hwh
@contact: ruc_hwh_2013@163.com
@file: batch_test.py
@time: 2018/2/8 上午11:39
"""

from src.train import train_model
from src.predict import predict
from conf.batch_test_conf import EXCEL_PATH, SAVE_PATH
import pandas as pd


"""
批量doc的测试，格式为excel，第一列名称，第二列歌词
"""

def batch_test(excel, save_path):
    mydata = pd.read_excel(excel)
    names = mydata.iloc[:, 0]
    print names
    lyrics = mydata.iloc[:, 1]
    print lyrics
    print mydata.shape[1]
    f = open(save_path,'w')
    for idx in range(mydata.shape[1]):
        f.write(names[idx].encode('utf-8') + '\t' + predict(lyrics[idx].encode('utf-8')) + '\n')
    f.close()

if __name__ == '__main__':
    batch_test(EXCEL_PATH, SAVE_PATH)


