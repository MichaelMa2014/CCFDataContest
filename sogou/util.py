# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import random
import re

import keras.preprocessing.sequence
import keras.preprocessing.text
import numpy
import pandas
import pynlpir

seed = 42


def init_random():
    random.seed(seed)
    numpy.random.seed(seed)


def raw_to_words(df, feature, stopwords=None, dictionary=None):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :param tuple|list|set|dict stopwords:
    :param tuple|list|set|dict dictionary:
    :rtype: pandas.DataFrame
    """
    # 标点符号替换成空格
    filter_char = keras.preprocessing.text.base_filter()
    df[feature] = df[feature].map(lambda stc: stc.translate({ord(c): ' ' for c in filter_char}))
    # 去除首尾空白字符
    df[feature] = df[feature].map(lambda stc: stc.strip())
    # 根据空白字符切分
    blank_re = re.compile(r'\s+')
    df[feature] = df[feature].map(blank_re.split)
    # 过滤空字符串
    df[feature] = df[feature].map(lambda wl: [w for w in wl if len(w)])
    # 过滤纯数字
    digit_re = re.compile(r'^\d+$')
    df[feature] = df[feature].map(lambda wl: filter(lambda w: not digit_re.match(w), wl))
    # 英文变成小写
    df[feature] = df[feature].map(lambda wl: [w.lower() for w in wl])
    # 分词
    df[feature] = df[feature].map(lambda wl: sum((pynlpir.segment(w, pos_tagging=False) for w in wl), []))
    # 再次过滤空字符串
    df[feature] = df[feature].map(lambda wl: [w for w in wl if len(w)])
    # 再次过滤纯数字
    df[feature] = df[feature].map(lambda wl: filter(lambda w: not digit_re.match(w), wl))
    # 去除停止词
    if stopwords:
        if isinstance(stopwords, (tuple, list)):
            stopwords = set(stopwords)
        df[feature] = df[feature].map(lambda wl: filter(lambda w: w not in stopwords, wl))
    # 只保留词典中存在的词
    if dictionary:
        if isinstance(dictionary, (tuple, list)):
            dictionary = set(dictionary)
        df[feature] = df[feature].map(lambda wl: filter(lambda w: w in dictionary, wl))
    return df


def raw_to_texts(df, feature, stopwords=None, dictionary=None):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :param tuple|list|set|dict stopwords:
    :param tuple|list|set|dict dictionary:
    :rtype: pandas.DataFrame
    """
    df = raw_to_words(df, feature, stopwords, dictionary)
    df[feature] = df[feature].map(lambda wl: ' '.join(wl))
    return df
