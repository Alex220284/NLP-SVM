#!/usr/bin/env python
# encoding: utf-8
"""
@author: hwh
@contact: ruc_hwh_2013@163.com
@file: crawl_conf.py
@time: 2018/2/7 下午12:05
"""

import os

# 存储根路径
DATA_SAVE_PATH = './data/'

# 歌曲页的目标链接，可自行添加不同类型歌曲链接和同一类型歌曲的爬取数量
liuxing_urls = {u'流行':
                  ['http://music.163.com/#/playlist?id=2045204807',
                   'http://music.163.com/#/playlist?id=2042006896',
                   'http://music.163.com/#/playlist?id=2055959723',
                   'http://music.163.com/#/playlist?id=2063351734',
                   'http://music.163.com/#/playlist?id=2049424729',
                   'http://music.163.com/#/playlist?id=2014095247',
                   'http://music.163.com/#/playlist?id=2023551932']
                }

gufeng_urls = {u'古风':
                   ['http://music.163.com/#/playlist?id=984166451',
                    'http://music.163.com/#/playlist?id=948471242',
                    'http://music.163.com/#/playlist?id=991478678',
                    'http://music.163.com/#/playlist?id=922791900'
                    ]}

destination_urls = [liuxing_urls, gufeng_urls]  # 目标url

use_pool = 1  # 1:使用多进程 0:不使用多进程

