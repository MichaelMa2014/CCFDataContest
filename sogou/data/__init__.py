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
english_stopwords = set(nltk.corpus.stopwords.words('english'))
chinese_stopwords = {word[:-1] for word in codecs.open('../util/stopwords.txt', 'rU', encoding='utf-8')}

stopwords = list(english_stopwords | chinese_stopwords | set(string.punctuation))

# 字段名
label_col = ['age', 'gender', 'education']


def load_train_data():
    """
    读入训练集数据
    """
    train_age = []
    train_gender = []
    train_education = []
    train_query = []
    with codecs.open('./data/user_tag_query.2W.TRAIN', encoding=conf.ENCODING) as train_file:
        for line in train_file:
            array = line.split('\t')
            train_age.append(int(array[1]))
            train_gender.append(int(array[2]))
            train_education.append(int(array[3]))
            train_query.append(array[4:])
    return pandas.DataFrame(
        {'age': train_age, 'gender': train_gender, 'education': train_education, 'query': train_query})


def load_test_data():
    """
    读入测试集数据
    """
    test_id = []
    test_query = []
    with codecs.open('./data/user_tag_query.2W.TEST', encoding=conf.ENCODING) as test_file:
        for line in test_file:
            array = line.split('\t')
            test_id.append(array[0])
            test_query.append(array[1:])
    return pandas.DataFrame({'id': test_id, 'query': test_query})


def process_data(df):
    """
    :param pandas.DataFrame df:
    :rtype: pandas.DataFrame
    """
    df['query'] = df['query'].map(lambda l: ' '.join(l))
    df = util.raw_to_texts(df, 'query')
    return df
