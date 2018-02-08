#!/usr/bin/env python
# encoding: utf-8
"""
@author: hwh
@contact: ruc_hwh_2013@163.com
@file: single_test.py
@time: 2018/2/8 上午11:34
"""

"""
单个doc的测试
"""

from src.train import train_model
from src.predict import predict


def single_test(doc):
    # 生成训练模型
    m = train_model()
    m.train_model()
    predict(doc)

if __name__ == '__main__':
    doc = """我爱你"""
    single_test(doc)