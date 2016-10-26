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
import pandas
import sklearn.model_selection

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

        token_indice = {v: k for k, v in enumerate(ngram_set, start=max_feature + 1)}
        sequences = _add_ngram(sequences, token_indice, ngram)

    return sequences


def transform(df, ngram):
    """
    转化成序列矩阵
    :param pandas.DataFrame df:
    :param int ngram:
    :rtype: pandas.DataFrame
    """
    tokenizer = feature.build_tokenizer(df)
    sequences = tokenizer.texts_to_sequences(line.encode('utf-8') for line in df['query'].values)
    sequences = _build_ngram_sequences(sequences, ngram)
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=2000 * ngram, padding='post',
                                                           truncating='post')
    print('sequences shape:', sequences.shape)

    df.drop('query', axis=1, inplace=True)
    return df.join(pandas.DataFrame(sequences.tolist()))


def build_train_set(label, validation_split=0.0, dummy=False, ngram=1):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool dummy: 是否将类别转化成哑变量
    :param int ngram:
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_train_df_{ngram}gram.hdf'.format(file_name=_file_name, ngram=ngram))
    if os.path.exists(path):
        train_df = pandas.read_hdf(path)
    else:
        train_df = data.load_train_data()
        data.process_data(train_df, remove_stopwords=False)
        train_df = transform(train_df, ngram)
        train_df.to_hdf(path, 'train_df')

    max_feature = train_df.max().max()

    # 去掉label未知的数据
    train_df = train_df[train_df[label] > 0]

    stratify = train_df[label].values
    train_df.drop(data.label_col, axis=1, inplace=True)
    target = keras.utils.np_utils.to_categorical(stratify) if dummy else stratify
    print('train_df shape:', train_df.shape)

    if validation_split == 0.0:
        return train_df.values, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_df.values, target,
                                                                              test_size=validation_split,
                                                                              random_state=util.seed,
                                                                              stratify=stratify)
    return X_train, y_train, X_val, y_val, max_feature


def build_test_set(ngram=1):
    """
    处理测试集
    :param int ngram:
    :rtype: numpy.ndarray
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_test_df_{ngram}gram.hdf'.format(file_name=_file_name, ngram=ngram))
    if os.path.exists(path):
        test_df = pandas.read_hdf(path)
    else:
        test_df = data.load_test_data()
        test_df.drop('id', axis=1, inplace=True)
        data.process_data(test_df, remove_stopwords=False)
        test_df = transform(test_df, ngram)
        test_df.to_hdf(path, 'train_df')

    print('test_df shape:', test_df.shape)
    return test_df.values
