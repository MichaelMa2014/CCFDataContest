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

import conf
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


def build_train(validation_split=0.0, sparse=True):
    """
    处理训练集和验证集
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool sparse:
    """
    if not os.path.exists(conf.TEMP_DIR):
        os.mkdir(conf.TEMP_DIR)
    path = os.path.abspath('{temp}/{file}_train.{format}'.format(
        temp=conf.TEMP_DIR, file=_file_name, format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_train('query')
        query = data.process_data(query, remove_stopwords=True)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(path)

    # 去掉label未知的数据
    label_rows = numpy.array(
        [data.load_train(label) > 0 for label in data.label_col])
    query = query[label_rows.all(axis=0)]

    stratify = label_rows[label_rows.all(axis=0)]
    target = []
    y_shape = []
    for col in range(stratify.shape[0]):
        dummy = keras.utils.np_utils.to_categorical(stratify[:, col])
        target.append(dummy)
        y_shape.append(dummy.shape[1])
    target = numpy.concatenate(target, axis=1)
    util.logger.info('train_df shape: {shape}'.format(shape=query.shape))

    if validation_split == 0.0:
        return query, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        query, target, test_size=validation_split, random_state=util.seed,
        stratify=stratify)
    return X_train, y_train, X_val, y_val, y_shape


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
        query = data.process_data(query, remove_stopwords=True)
        query = _transform(query, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(
        path, mmap_mode='r+')
    util.logger.info('test_df shape: {shape}'.format(shape=query.shape))
    return query
