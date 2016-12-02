# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import codecs
import string

import nltk
import numpy

import conf
import util

# 加载nltk.corpus.stopwords停用词
_english_stopwords = set(nltk.corpus.stopwords.words('english'))
_chinese_stopwords = {word[:-1] for word in
                      codecs.open('data/stopwords.txt', 'rU', encoding='utf-8')}
stopwords = list(
    _english_stopwords | _chinese_stopwords | set(string.punctuation))

# 数据地址
_train_path = 'data/user_tag_query.10W.TRAIN'
_test_path = 'data/user_tag_query.10W.TEST'

# 字段名
label_col = ['age', 'gender', 'education']


def load_train(col):
    """
    读入训练集数据
    :param str|unicode col:
    :rtype: numpy.ndarray
    """
    mapping = {'age': 1, 'gender': 2, 'education': 3}
    data = []
    with codecs.open(_train_path, encoding=conf.ENCODING) as train_file:
        for line in train_file:
            array = line.split('\t')
            if col == 'id':
                data.append(array[0])
            elif col == 'query':
                data.append(array[4:])
            else:
                data.append(int(array[mapping[col]]))
    return numpy.array(data)


def load_test(col):
    """
    读入测试集数据
    :param str|unicode col:
    :rtype: numpy.ndarray
    """
    data = []
    with codecs.open(_test_path, encoding=conf.ENCODING) as test_file:
        for line in test_file:
            array = line.split('\t')
            if col == 'id':
                data.append(array[0])
            else:
                data.append(array[1:])
    return numpy.array(data)


def process_data(rows, remove_stopwords):
    """
    :param numpy.ndarray rows:
    :param bool remove_stopwords:
    :rtype: numpy.ndarray
    """
    rows = util.umap(lambda l: ' '.join(l), rows)
    return util.rows_to_texts(rows, remove_stopwords)
