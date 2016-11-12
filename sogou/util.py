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
import numpy
import pandas
import pynlpir

import data

seed = 42

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s:%(msecs)05.1f pid:%(process)d [%(levelname)s] (%(funcName)s) %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()


def init_random():
    random.seed(seed)
    numpy.random.seed(seed)


def raw_to_words(df, feature, remove_stopwords=False, dictionary=None):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :param bool remove_stopwords:
    :param tuple|list|set|dict dictionary:
    """
    filter_char = keras.preprocessing.text.base_filter()
    blank_re = re.compile(r'\s+')
    digit_re = re.compile(r'^\d+$')
    df[feature] = (df[feature]
                   # 标点符号替换成空格
                   .map(lambda stc: stc.translate({ord(c): ' ' for c in filter_char}))
                   # 去除首尾空白字符
                   .map(lambda stc: stc.strip())
                   # 根据空白字符切分
                   .map(blank_re.split)
                   # 过滤空字符串
                   .map(lambda wl: [w for w in wl if len(w)])
                   # 过滤纯数字
                   .map(lambda wl: filter(lambda w: not digit_re.match(w), wl))
                   # 英文变成小写
                   .map(lambda wl: [w.lower() for w in wl])
                   # 分词
                   .map(lambda wl: sum((pynlpir.segment(w, pos_tagging=False) for w in wl), []))
                   # 再次过滤空字符串
                   .map(lambda wl: [w for w in wl if len(w)])
                   # 再次过滤纯数字
                   .map(lambda wl: filter(lambda w: not digit_re.match(w), wl)))
    # 去除停止词
    if remove_stopwords:
        df[feature] = df[feature].map(lambda wl: filter(lambda w: w not in data.stopwords, wl))
    # 只保留词典中存在的词
    if dictionary:
        if isinstance(dictionary, (tuple, list)):
            dictionary = set(dictionary)
        df[feature] = df[feature].map(lambda wl: filter(lambda w: w in dictionary, wl))


def raw_to_texts(df, feature, remove_stopwords=False, dictionary=None):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :param bool remove_stopwords:
    :param tuple|list|set|dict dictionary:
    """
    raw_to_words(df, feature, remove_stopwords, dictionary)
    df[feature] = df[feature].map(lambda wl: ' '.join(wl))
