# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import gensim
import keras.preprocessing.text
import keras.utils.np_utils
import numpy
import pandas

import data
import util

_tokenizer = None
_train_df = None
_test_df = None
_test_id = None

word_counts = None


def build_word2vec_model(word_vec_dim=300):
    """
    :param int word_vec_dim:
    :rtype: gensim.models.Word2Vec
    """
    src_df = data.load_train_data()
    text_df = pandas.DataFrame({'query': sum(src_df['query'].values, [])})
    text_df = util.raw_to_texts(text_df, 'query')
    print("text_df shape:", text_df.shape)

    model = gensim.models.Word2Vec(text_df['query'], size=word_vec_dim, min_count=1, workers=1, seed=util.seed)
    model.init_sims(replace=False)
    return model


def process(df, test=False):
    """
    文本处理部分
    :param pandas.DataFrame df:
    :param bool test:
    :rtype: pandas.DataFrame
    """
    # 分词与合并
    df.loc[:, 'query'] = df.loc[:, 'query'].map(lambda l: ' '.join(l))
    df = util.raw_to_texts(df, 'query')

    # 转化成序列矩阵
    global _tokenizer, word_counts
    if not test:
        _tokenizer = keras.preprocessing.text.Tokenizer()
        _tokenizer.fit_on_texts(line.encode('utf-8') for line in df['query'])
        if word_counts is None:
            word_counts = len(_tokenizer.word_index)

    sequences = util.texts_to_sequences(df, 'query', _tokenizer, maxlen=2000)
    df.drop('query', axis=1, inplace=True)
    return df.join(pandas.DataFrame(sequences.tolist()))


def build_train_set(label):
    """
    处理训练集
    :param str|unicode label: 类别标签
    """
    global _train_df
    if _train_df is None:
        _train_df = data.load_train_data()
        _train_df = process(_train_df)

    # 去掉label未知的数据
    train_df = _train_df[_train_df[label] > 0].copy()

    target = keras.utils.np_utils.to_categorical(train_df[label].values)
    train_df.drop(data.label_col, axis=1, inplace=True)
    print("train_df shape:", train_df.shape)
    print("target shape:", target.shape)

    return train_df.values, target


def build_test_set():
    """
    处理测试集
    """
    global _test_df, _test_id
    if _test_df is None or _test_id is None:
        _test_df = data.load_test_data()
        _test_df = process(_test_df, test=True)

        _test_id = _test_df['id']
        _test_df.drop('id', axis=1, inplace=True)
        print("test_df shape:", _test_df.shape)

    return _test_df.values, _test_id


def flush():
    """
    清空缓存
    """
    global _tokenizer, _train_df, _test_df, _test_id
    if _tokenizer is not None:
        _tokenizer = None
    if _train_df is not None:
        _train_df = None
    if _test_df is not None:
        _test_df = None
    if _test_id is not None:
        _test_id = None


def build_weights_matrix(word_vec_dim=300):
    """
    根据词向量构建初始化权重矩阵
    :param int word_vec_dim:
    :rtype: numpy.ndarray
    """
    word2vec = build_word2vec_model(word_vec_dim)

    global word_counts
    weights = numpy.zeros((word_counts + 1, word_vec_dim))
    for word, index in _tokenizer.word_index.items():
        # if index <= max_features and word in word2vec.vocab:
        #     weights[index, :] = word2vec[word]
        if word in word2vec.vocab:
            weights[index, :] = word2vec[word]
        else:
            weights[index, :] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)

    print("word2vec_weights shape:", weights.shape)
    return weights
