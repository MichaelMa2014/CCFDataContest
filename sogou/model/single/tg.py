# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import pandas
import sklearn.model_selection
import tgrocery

import data
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_train_df.hdf'.format(file_name=_file_name))
    if os.path.exists(path):
        train_df = pandas.read_hdf(path)
    else:
        train_df = data.load_train_data()
        data.process_data(train_df, remove_stopwords=True)
        train_df.to_hdf(path, 'train_df')

    train_df = train_df[train_df[label] > 0]
    target = train_df[label].astype('category')
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(train_df['query'].values, target,
                                                                              test_size=0.1, random_state=util.seed)

    clf = tgrocery.Grocery('sogou')
    clf.train(zip(y_train, X_train))

    val_acc = clf.test(zip(y_val, X_val)).accuracy_overall
    util.logger.info('val_acc: {acc}'.format(acc=val_acc))

    return clf, val_acc


def run():
    util.logger.info('TextGrocery')
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/{file_name}_test_df.hdf'.format(file_name=_file_name))
    if os.path.exists(path):
        test_df = pandas.read_hdf(path)
    else:
        test_df = data.load_test_data()
        test_df.drop('id', axis=1, inplace=True)
        data.process_data(test_df, remove_stopwords=True)
        test_df.to_hdf(path, 'test_df')

    pred_age = [clf_age.predict(text).predicted_y for text in test_df['query']]
    pred_gender = [clf_gender.predict(text).predicted_y for text in test_df['query']]
    pred_education = [clf_education.predict(text).predicted_y for text in test_df['query']]

    submissions.save_csv(pred_age, pred_gender, pred_education, '{file_name}.csv'.format(file_name=_file_name))
