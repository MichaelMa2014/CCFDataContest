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

import keras.preprocessing.text
import numpy
import pandas
import pynlpir
import sklearn.preprocessing

import conf
import data

random_state = 42
random.seed(random_state)
numpy.random.seed(random_state)


def process_dummy(df, feature):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :rtype: pandas.DataFrame
    """
    df = df.join(pandas.get_dummies(df.loc[:, feature], prefix=feature))
    df.drop(feature, axis=1, inplace=True)
    return df


def process_scale(df, feature):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :rtype: pandas.DataFrame
    """
    df.loc[:, feature].fillna(df.loc[:, feature].dropna().median(), inplace=True)
    df.loc[:, feature] = sklearn.preprocessing.scale(df.loc[:, feature].astype(numpy.float64), copy=False)
    return df


def text_split(df, feature, stopwords=None):
    """
    :param pandas.DataFrame df:
    :param str|unicode feature:
    :param list|set stopwords:
    :rtype: pandas.DataFrame
    """
    # 标点符号替换成空格
    filter_char = keras.preprocessing.text.base_filter()
    df.loc[:, feature] = df.loc[:, feature].map(lambda stc: stc.translate({ord(c): ' ' for c in filter_char}))
    # 去除首尾空白字符
    df.loc[:, feature] = df.loc[:, feature].map(lambda stc: stc.strip())
    # 根据空白字符切分
    blank_re = re.compile(r'\s+')
    df.loc[:, feature] = df.loc[:, feature].map(blank_re.split)
    # 过滤空字符串
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: [w for w in wl if len(w)])
    # 过滤纯数字
    digit_re = re.compile(r'^\d+$')
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: filter(lambda w: not digit_re.match(w), wl))
    # 英文变成小写
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: [w.lower() for w in wl])
    # 分词
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: sum((pynlpir.segment(w, pos_tagging=False) for w in wl), []))
    # 再次过滤空字符串
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: [w for w in wl if len(w)])
    # 再次过滤纯数字
    df.loc[:, feature] = df.loc[:, feature].map(lambda wl: filter(lambda w: not digit_re.match(w), wl))
    # 去除停止词
    if stopwords:
        df.loc[:, feature] = df.loc[:, feature].map(lambda wl: filter(lambda w: w not in stopwords, wl))
    return df


def text_merge(df, features, col_name):
    """
    :param pandas.DataFrame df:
    :param list features:
    :param str|unicode col_name:
    :rtype: pandas.DataFrame
    """
    words = df.apply(lambda wl: sum((wl[feature] for feature in features), []), axis=1)
    words.name = col_name
    words = words.map(lambda wl: ' '.join(wl))
    df = df.join(words)
    df.drop(features, axis=1, inplace=True)
    return df


def to_words(df, features, col_name='words'):
    """
    分词与合并
    :param pandas.DataFrame df:
    :param list features:
    :param str|unicode col_name:
    :rtype: pandas.DataFrame
    """
    df = df.loc[:, features]
    conf.pynlpir.init()
    for feature in features:
        df = text_split(df, feature, set(data.stopwords))
    df = text_merge(df, features, col_name)
    return df
