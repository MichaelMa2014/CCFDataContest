# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import pandas
import sklearn.feature_extraction
import sklearn.model_selection

import conf
import data
import util

_vectorizer = None
_train_df = None
_X_test = None
_test_id = None


def process(df, col_name='words', test=False):
    """
    文本处理部分
    :param pandas.DataFrame df:
    :param str|unicode col_name:
    :param bool test:
    :rtype: pandas.DataFrame
    """
    # 分词与合并
    conf.pynlpir.init()

    df.loc[:, 'query'] = df.loc[:, 'query'].map(lambda l: ' '.join(l))
    df = util.text_split(df, 'query')
    df = util.text_merge(df, ['query'], col_name)

    # 转化成词频向量
    global _vectorizer
    if not test:
        _vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=data.stopwords, max_features=5000)
        vec = _vectorizer.fit_transform(df[col_name]).toarray()
    else:
        assert _vectorizer is not None
        vec = _vectorizer.transform(df[col_name]).toarray()
    print("tf-idf shape:", vec.shape)

    df = df.join(pandas.DataFrame(vec))

    df.drop(col_name, axis=1, inplace=True)
    return df


def build_train_set(label, validation_split=0.0):
    """
    处理训练集和验证集
    :param str|unicode label: 类别标签
    :param float validation_split: 验证集比例，如果为0.0则不返回验证集
    """
    global _train_df
    if _train_df is None:
        _train_df = data.load_train_data()
        _train_df = process(_train_df)

    # 去掉label未知的数据
    train_df = _train_df[_train_df[label] > 0].copy()

    # 去掉与label无关的列
    train_df.drop([col for col in data.label_col if col != label], axis=1, inplace=True)

    # # 每种样本不重复抽取至多sample_num个并随机打乱
    # sample_num = 1000
    # train_df_sample = train_df[train_df[label] == 1]
    # train_df_sample = train_df_sample.sample(n=min(sample_num, train_df_sample.shape[0]),
    #                                          random_state=util.random_state)
    # for num in data.ranges[label]:
    #     if num > 1:
    #         tmp_df = train_df[train_df[label] == num]
    #         tmp_df = tmp_df.sample(n=min(sample_num, tmp_df.shape[0]), random_state=util.random_state)
    #         train_df_sample = train_df_sample.append(tmp_df)
    # train_df = train_df_sample.sample(frac=1, random_state=util.random_state).reset_index(drop=True)

    target = train_df[label].astype('category')
    train_df.drop(label, axis=1, inplace=True)
    print("train_df shape:", train_df.shape)

    if validation_split == 0.0:
        return train_df, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_df, target,
                                                                              test_size=validation_split,
                                                                              random_state=util.random_state)
    return X_train, y_train, X_val, y_val


def build_test_set():
    """
    处理测试集
    """
    global _X_test, _test_id
    if _X_test is None or _test_id is None:
        test_df = data.load_test_data()
        test_df = process(test_df, test=True)

        _test_id = test_df['id']
        test_df.drop('id', axis=1, inplace=True)
        print("test_df shape:", test_df.shape)

        _X_test = test_df

    return _X_test, _test_id


def flush():
    """
    清空训练集缓存
    """
    global _vectorizer, _train_df
    _vectorizer = None
    _train_df = None