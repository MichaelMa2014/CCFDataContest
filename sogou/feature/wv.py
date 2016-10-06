# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import gensim
import keras.preprocessing.sequence
import keras.preprocessing.text
import keras.utils.np_utils
import numpy
import pandas
import sklearn.externals.joblib
import sklearn.model_selection

import data
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build_tokenizer(df=None):
    """
    :param pandas.DataFrame df:
    :rtype: keras.preprocessing.text.Tokenizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_tokenizer.model'.format(file_name=_file_name))
    if os.path.exists(path):
        tokenizer = sklearn.externals.joblib.load(path)
    else:
        if df is None:
            df = data.load_train_data()
            df = data.process_data(df)

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(line.encode('utf-8') for line in df['query'])
        sklearn.externals.joblib.dump(tokenizer, path)

    return tokenizer


def transform(df):
    """
    文本处理部分
    :param pandas.DataFrame df:
    :rtype: pandas.DataFrame
    """
    # 转化成序列矩阵
    tokenizer = build_tokenizer(df)
    sequences = tokenizer.texts_to_sequences(line.encode('utf-8') for line in df['query'].values)
    sequences = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=2000, padding='post', truncating='post')
    print('sequences shape:', sequences.shape)

    df.drop('query', axis=1, inplace=True)
    return df.join(pandas.DataFrame(sequences.tolist()))


def build_train_set(label, validation_split=0.0, dummy=False):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    :param bool dummy: 是否将类别转化成哑变量
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_train_df.hdf'.format(file_name=_file_name))
    if os.path.exists(path):
        train_df = pandas.read_hdf(path)
    else:
        train_df = data.load_train_data()
        train_df = data.process_data(train_df)
        train_df = transform(train_df)
        train_df.to_hdf(path, 'train_df')

    # 去掉label未知的数据
    train_df = train_df[train_df[label] > 0]

    target = keras.utils.np_utils.to_categorical(train_df[label].values) if dummy else train_df[label].astype(
        'category')
    train_df.drop(data.label_col, axis=1, inplace=True)
    print('train_df shape:', train_df.shape)
    print('target shape:', target.shape)

    if validation_split == 0.0:
        return train_df.values, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_df.values, target,
                                                                              test_size=validation_split,
                                                                              random_state=util.seed)
    return X_train, y_train, X_val, y_val


def build_test_set():
    """
    处理测试集
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_test_df.hdf'.format(file_name=_file_name))
    if os.path.exists(path):
        test_df = pandas.read_hdf(path)
    else:
        test_df = data.load_test_data()
        test_df = data.process_data(test_df)
        test_df = transform(test_df)
        test_df.to_hdf(path, 'train_df')

    test_id = test_df['id']
    test_df.drop('id', axis=1, inplace=True)
    print('test_df shape:', test_df.shape)

    return test_df.values, test_id


def build_word2vec_model(word_vec_dim):
    """
    :param int word_vec_dim:
    :rtype: gensim.models.Word2Vec
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_word2vec_dim{dim}.model'.format(file_name=_file_name, dim=word_vec_dim))
    if os.path.exists(path):
        model = gensim.models.Word2Vec.load(path)
    else:
        src_df = data.load_train_data()
        text_df = pandas.DataFrame({'query': sum(src_df['query'].values, [])})
        text_df = util.raw_to_texts(text_df, 'query')
        print('text_df shape:', text_df.shape)

        model = gensim.models.Word2Vec(text_df['query'], size=word_vec_dim, min_count=1, workers=1, seed=util.seed)
        model.init_sims(replace=False)
        model.save(path)

    return model


def build_weights_matrix(word_vec_dim=300):
    """
    根据词向量构建初始化权重矩阵
    :param int word_vec_dim:
    :rtype: numpy.ndarray
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath(
        'temp/{file_name}_weights_dim{dim}.npy'.format(file_name=_file_name, dim=word_vec_dim))
    if os.path.exists(path):
        weights = numpy.load(path)
    else:
        word2vec = build_word2vec_model(word_vec_dim)

        tokenizer = build_tokenizer()

        weights = numpy.zeros((get_max_feature() + 1, word_vec_dim))
        for word, index in tokenizer.word_index.items():
            if word in word2vec.vocab:
                weights[index] = word2vec[word]
            else:
                weights[index] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)

        print('word2vec_weights shape:', weights.shape)
        numpy.save(path, weights)

    return weights


def get_max_feature():
    tokenizer = build_tokenizer()
    return numpy.max(tokenizer.word_index.values())
