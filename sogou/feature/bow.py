# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import keras.utils.np_utils
import numpy
import scipy.io
import scipy.sparse
import sklearn.model_selection

import data
import feature
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def _transform(rows, sparse):
    """
    转化成词频向量
    :param numpy.ndarray rows:
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    vectorizer = feature.build_vectorizer(rows)
    vectors = vectorizer.transform(rows)
    util.logger.info(
        'tf-idf vectors shape: {shape}'.format(shape=vectors.shape))
    return vectors if sparse else vectors.toarray()


def build_train(label, validation_split=0.0, dummy=False, sparse=True):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool dummy: 是否将类别转化成哑变量
    :param bool sparse:
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath(
        'temp/{file}_train.{format}'.format(file=_file_name,
                                            format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_train('query')
        query = data.process_data(query, remove_stopwords=True)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(path,
                                                                    mmap_mode='r+')

    # 去掉label未知的数据
    label_rows = data.load_train(label)
    query = query[label_rows > 0]

    stratify = label_rows[label_rows > 0]
    target = keras.utils.np_utils.to_categorical(
        stratify) if dummy else stratify
    util.logger.info('train_df shape: {shape}'.format(shape=query.shape))

    if validation_split == 0.0:
        return query, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        query, target, test_size=validation_split, random_state=util.seed,
        stratify=stratify)
    return X_train, y_train, X_val, y_val


def build_test(sparse=True):
    """
    处理测试集
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath(
        'temp/{file}_train.{format}'.format(file=_file_name,
                                            format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_test('query')
        query = data.process_data(query, remove_stopwords=True)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(path,
                                                                    mmap_mode='r+')
    util.logger.info('test_df shape: {shape}'.format(shape=query.shape))
    return query
