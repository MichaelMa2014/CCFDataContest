# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import keras.preprocessing.sequence
import keras.utils.np_utils
import numpy
import scipy.io
import scipy.sparse
import sklearn.model_selection

import conf
import data
import feature
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def _transform(rows, sparse=True):
    """
    转化成序列矩阵
    :param numpy.ndarray rows:
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    tokenizer = feature.build_tokenizer(rows)
    sequences = tokenizer.texts_to_sequences(
        line.encode('utf-8') for line in rows)
    sequences = keras.preprocessing.sequence.pad_sequences(sequences,
                                                           maxlen=2000,
                                                           padding='post',
                                                           truncating='post')
    util.logger.info('sequences shape: {shape}'.format(shape=sequences.shape))
    return scipy.sparse.csr_matrix(sequences) if sparse else sequences


def build_train(label, validation_split=0.0, dummy=False, sparse=True):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool dummy: 是否将类别转化成哑变量
    :param bool sparse:
    """
    if not os.path.exists(conf.TEMP_DIR):
        os.mkdir(conf.TEMP_DIR)
    path = os.path.abspath('{temp}/{file}_train.{format}'.format(
        temp=conf.TEMP_DIR, file=_file_name, format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_train('query')
        query = data.process_data(query, remove_stopwords=False)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(path)

    max_feature = query.max()

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
    return X_train, y_train, X_val, y_val, max_feature


def build_test(sparse=True):
    """
    处理测试集
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    if not os.path.exists(conf.TEMP_DIR):
        os.mkdir(conf.TEMP_DIR)
    path = os.path.abspath('{temp}/{file}_test.{format}'.format(
        temp=conf.TEMP_DIR, file=_file_name, format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_test('query')
        query = data.process_data(query, remove_stopwords=False)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(path)
    util.logger.info('test_df shape: {shape}'.format(shape=query.shape))
    return query
