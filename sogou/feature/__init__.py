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
import pandas
import sklearn.externals.joblib
import sklearn.feature_extraction.text

import data
import util


def build_vectorizer(df=None):
    """
    :param pandas.DataFrame df:
    :retype: sklearn.feature_extraction.text.TfidfVectorizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/vectorizer.model')
    if os.path.exists(path):
        vectorizer = sklearn.externals.joblib.load(path)
    else:
        if df is None:
            df = data.load_train_data()
            df = data.process_data(df, remove_stopwords=True)

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=data.stopwords, max_features=8000)
        vectorizer.fit(df['query'])
        sklearn.externals.joblib.dump(vectorizer, path)

    return vectorizer


def build_tokenizer(df=None):
    """
    :param pandas.DataFrame df:
    :rtype: keras.preprocessing.text.Tokenizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/tokenizer.model')
    if os.path.exists(path):
        tokenizer = sklearn.externals.joblib.load(path)
    else:
        if df is None:
            df = data.load_train_data()
            df = data.process_data(df, remove_stopwords=False)

        tokenizer = keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(line.encode('utf-8') for line in df['query'])
        sklearn.externals.joblib.dump(tokenizer, path)

    return tokenizer


def build_word2vec_model(word_vec_dim):
    """
    :param int word_vec_dim:
    :rtype: gensim.models.Word2Vec
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/word2vec_dim{dim}.model'.format(dim=word_vec_dim))
    if os.path.exists(path):
        model = gensim.models.Word2Vec.load(path)
    else:
        src_df = data.load_train_data()
        text_df = pandas.DataFrame({'query': sum(src_df['query'].values, [])})
        text_df = util.raw_to_texts(text_df, 'query', remove_stopwords=False)
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
        'temp/weights_dim{dim}.npy'.format(dim=word_vec_dim))
    if os.path.exists(path):
        weights = numpy.load(path)
    else:
        word2vec = build_word2vec_model(word_vec_dim)

        tokenizer = build_tokenizer()
        max_feature = numpy.max(tokenizer.word_index.values())

        weights = numpy.zeros((max_feature + 1, word_vec_dim))
        for word, index in tokenizer.word_index.items():
            if word in word2vec.vocab:
                weights[index] = word2vec[word]
            else:
                weights[index] = numpy.random.uniform(-0.25, 0.25, word_vec_dim)

        print('word2vec_weights shape:', weights.shape)
        numpy.save(path, weights)

    return weights
