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


def _build_ngram_set(sequence, ngram):
    """
    Extract a set of n-grams from a list of integers.
    >>> _build_ngram_set([1, 4, 9, 4, 1, 4], ngram=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> _build_ngram_set([1, 4, 9, 4, 1, 4], ngram=3)
    {(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)}
    :param list sequence:
    :param int ngram:
    :rtype: set
    """
    return set(zip(*[sequence[i:] for i in range(ngram)]))


def _add_ngram(sequences, token_indice, ngram):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> _add_ngram(sequences, token_indice, ngram=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> _add_ngram(sequences, token_indice, ngram=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    :param list sequences:
    :param dict token_indice:
    :param int ngram:
    :rtype: list
    """
    for seq in sequences:
        for i in range(len(seq) - ngram + 1):
            for n in range(2, ngram + 1):
                token = tuple(seq[i:i + n])
                if token in token_indice:
                    seq.append(token_indice[token])
    return sequences


def _build_ngram_sequences(sequences, ngram):
    """
    :param list sequences:
    :param int ngram:
    :rtype: list
    """
    if ngram > 1:
        ngram_set = set()
        for seq in sequences:
            for n in range(2, ngram + 1):
                ngram_set.update(_build_ngram_set(seq, ngram=n))

        tokenizer = feature.build_tokenizer()
        max_feature = numpy.max(tokenizer.word_index.values())

        token_indice = {v: k for k, v in
                        enumerate(ngram_set, start=max_feature + 1)}
        sequences = _add_ngram(sequences, token_indice, ngram)

    return sequences


def _transform(rows, ngram, sparse=True):
    """
    转化成序列矩阵
    :param numpy.ndarray rows:
    :param int ngram:
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    tokenizer = feature.build_tokenizer(rows)
    sequences = tokenizer.texts_to_sequences(
        line.encode('utf-8') for line in rows)
    sequences = _build_ngram_sequences(sequences, ngram)
    print('mean:', numpy.mean([len(x) for x in sequences]))
    print('std:', numpy.std([len(x) for x in sequences]))
    print('median:', numpy.median([len(x) for x in sequences]))
    print('max:', numpy.max([len(x) for x in sequences]))
    sequences = keras.preprocessing.sequence.pad_sequences(sequences,
                                                           maxlen=2000 * ngram,
                                                           padding='post',
                                                           truncating='post')
    util.logger.info('sequences shape: {shape}'.format(shape=sequences.shape))
    return scipy.sparse.csr_matrix(sequences) if sparse else sequences


def build_train(label, validation_split=0.0, dummy=False, ngram=1, sparse=True):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool dummy: 是否将类别转化成哑变量
    :param bool sparse:
    :param int ngram:
    """
    if not os.path.exists(conf.TEMP_DIR):
        os.mkdir(conf.TEMP_DIR)
    path = os.path.abspath('{temp}/{file}_train_{n}gram.{format}'.format(
        temp=conf.TEMP_DIR, file=_file_name, n=ngram,
        format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_train('query')
        query = data.process_data(query, remove_stopwords=False)
        query = _transform(query, ngram, sparse)
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


def build_test(ngram=1, sparse=True):
    """
    处理测试集
    :param int ngram:
    :param bool sparse:
    :rtype: scipy.sparse.spmatrix|numpy.ndarray
    """
    if not os.path.exists(conf.TEMP_DIR):
        os.mkdir(conf.TEMP_DIR)
    path = os.path.abspath('{temp}/{file}_test_{n}gram.{format}'.format(
        temp=conf.TEMP_DIR, file=_file_name, n=ngram,
        format='mtx' if sparse else 'npy'))
    if not os.path.exists(path):
        query = data.load_test('query')
        query = data.process_data(query, remove_stopwords=False)
        query = _transform(query, ngram, sparse)
        if sparse:
            scipy.io.mmwrite(path, query)
        else:
            numpy.save(path, query)

    query = scipy.io.mmread(path).tocsr() if sparse else numpy.load(
        path, mmap_mode='r+')
    util.logger.info('test_df shape: {shape}'.format(shape=query.shape))
    return query
