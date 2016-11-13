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
import pandas

import conf
import util

# 加载nltk.corpus.stopwords停用词
_english_stopwords = set(nltk.corpus.stopwords.words('english'))
_chinese_stopwords = {word[:-1] for word in codecs.open('../util/stopwords.txt', 'rU', encoding='utf-8')}
stopwords = list(_english_stopwords | _chinese_stopwords | set(string.punctuation))

# 数据地址
_train_path = 'data/user_tag_query.2W.TRAIN'
_test_path = 'data/user_tag_query.2W.TEST'

# 字段名
label_col = ['age', 'gender', 'education']


def load_train_data():
    """
    读入训练集数据
    :rtype: pandas.DataFrame
    """
    data = {}
    for col in ('id', 'age', 'gender', 'education', 'query'):
        data[col] = []
    with codecs.open(_train_path, encoding=conf.ENCODING) as train_file:
        for line in train_file:
            array = line.split('\t')
            data['id'].append(array[0])
            data['age'].append(int(array[1]))
            data['gender'].append(int(array[2]))
            data['education'].append(int(array[3]))
            data['query'].append(array[4:])
    return pandas.DataFrame(data)


def load_test_data():
    """
    读入测试集数据
    :rtype: pandas.DataFrame
    """
    data = {}
    for col in ('id', 'query'):
        data[col] = []
    with codecs.open(_test_path, encoding=conf.ENCODING) as test_file:
        for line in test_file:
            array = line.split('\t')
            data['id'].append(array[0])
            data['query'].append(array[1:])
    return pandas.DataFrame(data)


def get_test_id():
    """
    :rtype: list
    """
    ids = []
    with codecs.open(_test_path, encoding=conf.ENCODING) as test_file:
        for line in test_file:
            array = line.split('\t')
            ids.append(array[0])
    return ids


def process_data(df, remove_stopwords):
    """
    :param pandas.DataFrame df:
    :param bool remove_stopwords:
    """
    df['query'] = df['query'].map(lambda l: ' '.join(l))
    util.raw_to_texts(df, 'query', remove_stopwords)


def load_split_train_data():
    """
    读入训练集数据并拆分
    :rtype: pandas.DataFrame
    """
    df = load_train_data()
    data = []
    for idx, row in df.iterrows():
        for query in row['query']:
            data.append({'age': row['age'], 'gender': row['gender'], 'education': row['education'], 'query': query})
    return pandas.DataFrame(data)


def load_split_test_data():
    """
    读入测试集数据并拆分
    :rtype: pandas.DataFrame
    """
    df = load_test_data()
    data = []
    for idx, row in df.iterrows():
        for query in row['query']:
            data.append({'id': row['id'], 'query': query})
    return pandas.DataFrame(data)


def process_split_data(df, remove_stopwords):
    """
    :param pandas.DataFrame df:
    :param bool remove_stopwords:
    """
    util.raw_to_texts(df, 'query', remove_stopwords)
