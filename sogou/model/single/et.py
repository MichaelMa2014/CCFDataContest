# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import sklearn.ensemble

import feature.bow
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build_clf():
    return sklearn.ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=util.seed)


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, validation_split=0.1)

    clf = build_clf()
    clf.fit(X_train, y_train)

    val_acc = clf.score(X_val, y_val)
    print('val_acc:', val_acc)

    return clf, val_acc


def run():
    print("ExtraTrees")
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    print('acc_final:', acc_final)

    X_test = feature.bow.build_test_set()

    pred_age = clf_age.predict(X_test)
    pred_gender = clf_gender.predict(X_test)
    pred_education = clf_education.predict(X_test)

    submissions.save_csv(pred_age, pred_gender, pred_education, '{file_name}.csv'.format(file_name=_file_name))
