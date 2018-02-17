#!/usr/bin/env python
# encoding: utf-8
"""
@author: hwh
@contact: ruc_hwh_2013@163.com
@file: logger_conf.py
@time: 2018/2/17 下午10:57
"""

import os
import logging.config

save_path = '../log/'

if os.path.exists(save_path) is not True:
    os.mkdir(save_path)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,  # 定义输出到文件的log级别，
    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',  # 定义输出log的格式
    datefmt='%Y-%m-%d %A %H:%M:%S',  # 时间
    filename=save_path + 'run.log',  # log文件名
    filemode='a')
