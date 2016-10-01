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


def process(df):
    # 分词与合并
    conf.pynlpir.init()

    df.loc[:, 'query'] = df.loc[:, 'query'].map(lambda l: ' '.join(l))
    df = util.text_split(df, 'query')
    col_name = 'words'
    df = util.text_merge(df, ['query'], col_name)

    # 转化成词频向量
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words=data.stopwords, min_df=2, max_features=5000)
    vec = vectorizer.fit_transform(df[col_name]).toarray()
    print(vec.shape)
    df = df.join(pandas.DataFrame(vec))
    df.drop(col_name, axis=1, inplace=True)
    return df


def get_train(label):
    train_df = data.load_train()

    # 去掉与label无关的列
    for col in data.label_col:
        if col != label:
            train_df.drop(col, axis=1, inplace=True)

    # 去掉label未知的数据
    train_df = train_df[train_df[label] > 0]

    # 每种样本不重复抽取至多sample_num个并随机打乱
    sample_num = 2000
    train_df_sample = train_df[train_df[label] == 1]
    train_df_sample = train_df_sample.sample(n=min(sample_num, train_df_sample.shape[0]),
                                             random_state=util.random_state)
    for num in data.ranges[label]:
        if num > 1:
            tmp_df = train_df[train_df[label] == num]
            df = tmp_df.sample(n=min(sample_num, tmp_df.shape[0]), random_state=util.random_state)
            train_df_sample = train_df_sample.append(df)
    train_df_sample = train_df_sample.sample(frac=1, random_state=util.random_state).reset_index(drop=True)

    train_df_sample = process(train_df_sample)

    # 分离出label
    le = sklearn.preprocessing.LabelEncoder()
    train_df_sample[label] = le.fit_transform(train_df_sample[label])
    target = train_df_sample[label].astype('category')
    train_df_sample.drop(label, axis=1, inplace=True)

    X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(train_df_sample.values,
                                                                                            target,
                                                                                            test_size=0.2,
                                                                                            random_state=util.random_state)
    return X_train, y_train, X_validation, y_validation


def get_test():
    test_df = data.load_test()

    test_df = process(test_df)

    X_test = test_df['query'].values
    test_id = test_df['id'].values

    return X_test, test_id
