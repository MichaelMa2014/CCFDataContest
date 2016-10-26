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
import model.single
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, validation_split=0.1)

    clfs = [
        ('bnb', model.single.bnb.build_clf()),
        ('et', model.single.et.build_clf()),
        ('lr', model.single.lr.build_clf()),
        ('mlp', model.single.mlp_sklearn.build_clf()),
        ('mnb', model.single.mnb.build_clf()),
        ('rf', model.single.rf.build_clf()),
        ('svm', model.single.svm.build_clf()),
        ('xgb', model.single.xgb.build_clf())
    ]

    clf = sklearn.ensemble.VotingClassifier(clfs, voting='hard')
    clf.fit(X_train, y_train)

    val_acc = clf.score(X_val, y_val)
    print('val_acc:', val_acc)

    return clf, val_acc


def run():
    print("Voting Ensemble (hard) with 8 models")
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
