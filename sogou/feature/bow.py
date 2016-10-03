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

import data
import util

_vectorizer = None
_train_df = None
_test_df = None
_test_id = None


def process(df, test=False):
    """
    文本处理部分
    :param pandas.DataFrame df:
    :param bool test:
    :rtype: pandas.DataFrame
    """
    df.loc[:, 'query'] = df.loc[:, 'query'].map(lambda l: ' '.join(l))
    df = util.raw_to_texts(df, 'query')

    # 转化成词频向量
    global _vectorizer
    if not test:
        _vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=data.stopwords, max_features=8000)
        vec = _vectorizer.fit_transform(df['query']).toarray()
    else:
        assert _vectorizer is not None
        vec = _vectorizer.transform(df['query']).toarray()
    print("tf-idf shape:", vec.shape)

    df.drop('query', axis=1, inplace=True)
    return df.join(pandas.DataFrame(vec))


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

    target = train_df[label].astype('category')
    train_df.drop(data.label_col, axis=1, inplace=True)
    print("train_df shape:", train_df.shape)

    if validation_split == 0.0:
        return train_df, target
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_df, target,
                                                                              test_size=validation_split,
                                                                              random_state=util.seed)
    return X_train, y_train, X_val, y_val


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

    return _test_df, _test_id


def flush():
    """
    清空缓存
    """
    global _vectorizer, _train_df, _test_df, _test_id
    if _vectorizer is not None:
        _vectorizer = None
    if _train_df is not None:
        _train_df = None
    if _test_df is not None:
        _test_df = None
    if _test_id is not None:
        _test_id = None
