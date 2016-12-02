# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import logging
import random
import re

import keras.preprocessing.sequence
import keras.preprocessing.text
import jieba
import numpy

import data

seed = 42

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s:%(msecs)05.1f pid:%(process)d [%(levelname)s] (%(funcName)s) %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def init_random():
    random.seed(seed)
    numpy.random.seed(seed)


def umap(func, data):
    return numpy.frompyfunc(func, 1, 1)(data)


def rows_to_words(rows, remove_stopwords=False, dictionary=None):
    """
    :param numpy.ndarray rows:
    :param bool remove_stopwords:
    :param tuple|list|set|dict dictionary:
    :rtype: numpy.ndarray
    """
    filter_char = keras.preprocessing.text.base_filter()
    blank_re = re.compile(r'\s+')
    digit_re = re.compile(r'^\d+$')

    # 标点符号替换成空格
    rows = umap(lambda stc: stc.translate({ord(c): ' ' for c in filter_char}),
                rows)
    # 去除首尾空白字符
    rows = umap(lambda stc: stc.strip(), rows)
    # 根据空白字符切分
    rows = umap(blank_re.split, rows)
    # 过滤空字符串
    rows = umap(lambda wl: [w for w in wl if len(w)], rows)
    # 过滤纯数字
    rows = umap(lambda wl: filter(lambda w: not digit_re.match(w), wl), rows)
    # 英文变成小写
    rows = umap(lambda wl: [w.lower() for w in wl], rows)
    # 分词
    rows = umap(lambda wl: [t for w in wl for t in jieba.cut_for_search(w)],
                rows)
    # 再次过滤空字符串
    rows = umap(lambda wl: [w for w in wl if len(w)], rows)
    # 再次过滤纯数字
    rows = umap(lambda wl: filter(lambda w: not digit_re.match(w), wl), rows)
    # 去除停止词
    if remove_stopwords:
        rows = umap(lambda wl: filter(lambda w: w not in data.stopwords, wl),
                    rows)
    # 只保留词典中存在的词
    if dictionary:
        if isinstance(dictionary, (tuple, list)):
            dictionary = set(dictionary)
        rows = umap(lambda wl: filter(lambda w: w in dictionary, wl), rows)
    return rows


def rows_to_texts(rows, remove_stopwords=False, dictionary=None):
    """
    :param numpy.ndarray rows:
    :param bool remove_stopwords:
    :param tuple|list|set|dict dictionary:
    :rtype: numpy.ndarray
    """
    rows = rows_to_words(rows, remove_stopwords, dictionary)
    return umap(lambda wl: ' '.join(wl), rows)
