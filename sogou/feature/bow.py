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
import pandas
import sklearn.externals.joblib
import sklearn.feature_extraction
import sklearn.model_selection

import data
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build_vectorizer(df=None):
    """
    :param pandas.DataFrame df:
    :retype: sklearn.feature_extraction.text.TfidfVectorizer
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_vectorizer.model'.format(file_name=_file_name))
    if os.path.exists(path):
        vectorizer = sklearn.externals.joblib.load(path)
    else:
        if df is None:
            df = data.load_train_data()
            df = data.process_data(df)

        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=data.stopwords, max_features=8000)
        vectorizer.fit(df['query'])
        sklearn.externals.joblib.dump(vectorizer, path)

    return vectorizer


def transform(df):
    """
    文本处理部分
    :param pandas.DataFrame df:
    :rtype: pandas.DataFrame
    """
    # 转化成词频向量
    vectorizer = build_vectorizer(df)
    vectors = vectorizer.transform(df['query']).toarray()
    print('tf-idf vectors shape:', vectors.shape)

    df.drop('query', axis=1, inplace=True)
    return df.join(pandas.DataFrame(vectors))


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

    # # 每种样本不重复抽取至多sample_num个并随机打乱
    # sample_num = 1000
    # train_df_sample = train_df[train_df[label] == 1]
    # train_df_sample = train_df_sample.sample(n=min(sample_num, train_df_sample.shape[0]), random_state=util.seed)
    # for num in data.ranges[label]:
    #     if num > 1:
    #         tmp_df = train_df[train_df[label] == num]
    #         tmp_df = tmp_df.sample(n=min(sample_num, tmp_df.shape[0]), random_state=util.seed)
    #         train_df_sample = train_df_sample.append(tmp_df)
    # train_df = train_df_sample.sample(frac=1, random_state=util.seed).reset_index(drop=True)

    target = keras.utils.np_utils.to_categorical(train_df[label].values) if dummy else train_df[label].astype(
        'category')
    train_df.drop(data.label_col, axis=1, inplace=True)
    print('train_df shape:', train_df.shape)

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
