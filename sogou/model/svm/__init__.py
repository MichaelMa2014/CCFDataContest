# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import sklearn.svm

import feature.bow
import submissions


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, 0.1)

    clf = sklearn.svm.LinearSVC()
    clf.fit(X_train, y_train)

    val_result = clf.score(X_val, y_val)
    print(val_result)

    return clf, val_result


def run():
    clf_age, val_age = build('age')
    clf_gender, val_gender = build('gender')
    clf_education, val_education = build('education')

    val_final = (val_age + val_gender + val_education) / 3
    print(val_final)

    X_test, test_id = feature.bow.build_test_set()

    pred_age = clf_age.predict(X_test)
    pred_gender = clf_gender.predict(X_test)
    pred_education = clf_education.predict(X_test)

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education, 'svm.csv')
