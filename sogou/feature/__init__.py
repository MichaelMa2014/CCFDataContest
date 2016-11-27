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
import keras.preprocessing.text
import numpy
import sklearn.externals.joblib
import sklearn.feature_extraction.text

import data
import util


def build_vectorizer(rows=None):
    """
    :param numpy.ndarray rows:
    :retype: sklearn.feature_extraction.text.TfidfVectorizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/vectorizer.model')
    if not os.path.exists(path):
        if rows is None:
            rows = data.load_train('query')
            rows = data.process_data(rows, remove_stopwords=True)

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            stop_words=data.stopwords, max_features=8000)
        vectorizer.fit(rows)
        sklearn.externals.joblib.dump(vectorizer, path, compress=True)

    vectorizer = sklearn.externals.joblib.load(path)
    return vectorizer


def build_tokenizer(rows=None):
    """
    :param numpy.ndarray rows:
    :rtype: keras.preprocessing.text.Tokenizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/tokenizer.model')
    if not os.path.exists(path):
        if rows is None:
            rows = data.load_train('query')
            rows = data.process_data(rows, remove_stopwords=True)

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(line.encode('utf-8') for line in rows)
        sklearn.externals.joblib.dump(tokenizer, path, compress=True)

    tokenizer = sklearn.externals.joblib.load(path)
    return tokenizer


def build_word2vec_model(word_vec_dim):
    """
    :param int word_vec_dim:
    :rtype: gensim.models.Word2Vec
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath(
        'temp/word2vec_dim{dim}.model'.format(dim=word_vec_dim))
    if not os.path.exists(path):
        text_path = os.path.abspath('temp/text_df.hdf')
        if not os.path.exists(text_path):
            query_sum = data.load_train('query').sum()
            util.rows_to_words(query_sum, remove_stopwords=False)
            numpy.save(text_path, query_sum)

        query_sum = numpy.load(text_path, mmap_mode='r+')
        util.logger.info('text_df shape: {shape}'.format(shape=query_sum.shape))

        assert word_vec_dim == 100 or word_vec_dim == 200 or word_vec_dim == 300
        if word_vec_dim == 100:
            # 100
            # (log_accuracy) capital-common-countries: 6.4% (11/171)
            # (log_accuracy) city-in-state: 52.0% (91/175)
            # (log_accuracy) family: 47.7% (63/132)
            # (log_accuracy) total: 34.5% (165/478)
            model = gensim.models.Word2Vec(query_sum, size=word_vec_dim,
                                           min_count=1, sample=1e-4, workers=1,
                                           seed=util.seed, iter=20)
        else:
            # 200
            # (log_accuracy) capital-common-countries: 7.6% (13/171)
            # (log_accuracy) city-in-state: 46.9% (82/175)
            # (log_accuracy) family: 52.3% (69/132)
            # (log_accuracy) total: 34.3% (164/478)

            # 300
            # (log_accuracy) capital-common-countries: 8.2% (14/171)
            # (log_accuracy) city-in-state: 51.4% (90/175)
            # (log_accuracy) family: 50.8% (67/132)
            # (log_accuracy) total: 35.8% (171/478)
            model = gensim.models.Word2Vec(query_sum, size=word_vec_dim,
                                           min_count=1, sample=1e-4, workers=1,
                                           seed=util.seed, iter=15)
        model.init_sims(replace=True)
        model.save(path)

    model = gensim.models.Word2Vec.load(path)
    # model.accuracy('data/word2vec_cn_accuracy.txt')
    util.init_random()
    return model


def build_weights_matrix(word_vec_dim):
    """
    根据词向量构建初始化权重矩阵
    :param int word_vec_dim:
    :rtype: numpy.ndarray
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/weights_dim{dim}.npy'.format(dim=word_vec_dim))
    if not os.path.exists(path):
        word2vec = build_word2vec_model(word_vec_dim)

        tokenizer = build_tokenizer()
        max_feature = numpy.max(tokenizer.word_index.values())

        weights = numpy.zeros((max_feature + 1, word_vec_dim))
        for word, index in tokenizer.word_index.items():
            if word in word2vec.vocab:
                weights[index] = word2vec[word]
            else:
                weights[index] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)

        util.logger.info(
            'word2vec_weights shape: {shape}'.format(shape=weights.shape))
        numpy.save(path, weights)
        del tokenizer
        del weights

    weights = numpy.load(path, mmap_mode='r+')
    util.init_random()
    return weights
