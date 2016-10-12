# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import numpy
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model
import sklearn.naive_bayes
import xgboost

import feature.bow
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build_blend_and_pred(label, X_test, n_folds):
    """
    构建分类器
    :param str|unicode label: 类别标签
    :param numpy.ndarray X_test:
    :param int n_folds:
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, validation_split=0.1)
    cls_cnt = len(numpy.unique(numpy.append(y_train, y_val)))
    print("classes count:", cls_cnt)

    skf = sklearn.model_selection.StratifiedKFold(n_folds, shuffle=True, random_state=util.seed)
    clfs = [
        sklearn.ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=util.seed),
        sklearn.linear_model.LogisticRegression(n_jobs=-1, random_state=util.seed),
        sklearn.naive_bayes.MultinomialNB(),
        sklearn.ensemble.RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=util.seed),
        xgboost.XGBClassifier(seed=util.seed)
    ]

    param_base = {'input_dim': X_train.shape[1], 'output_dim': cls_cnt + 1, 'shuffle': True}
    param = {
        'mlp': {'batch_size': model.single.mlp.param['batch_size'], 'nb_epoch': model.single.mlp.param[label]}
    }
    for c in param:
        param[c].update(param_base)
    clfs += [
        keras.wrappers.scikit_learn.KerasClassifier(model.single.mlp.build_clf, **param['mlp'])
    ]

    clfs_cnt = len(clfs)
    print("classifiers count:", clfs_cnt)

    blend_train = numpy.zeros((X_train.shape[0], clfs_cnt * cls_cnt))
    blend_val = numpy.zeros((X_val.shape[0], clfs_cnt * cls_cnt))
    blend_test = numpy.zeros((X_test.shape[0], clfs_cnt * cls_cnt))

    for i, clf in enumerate(clfs):
        print("classifier No.{i}:".format(i=i + 1), clf)
        idx = i * cls_cnt

        for j, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            print("fold:", j + 1)

            fold_X, fold_y = X_train[train_idx], y_train[train_idx]
            fold_test = X_train[test_idx]

            if isinstance(clf, keras.wrappers.scikit_learn.KerasClassifier):
                fold_y = keras.utils.np_utils.to_categorical(fold_y)

            clf.fit(fold_X, fold_y)
            fold_pred = clf.predict_proba(fold_test)

            if isinstance(clf, keras.wrappers.scikit_learn.KerasClassifier):
                fold_pred = numpy.delete(fold_pred, 0, axis=1)

            assert blend_train.shape[1] == fold_pred.shape[1] * clfs_cnt
            blend_train[test_idx, idx:idx + cls_cnt] = fold_pred

            pred_val = clf.predict_proba(X_val)
            pred_test = clf.predict_proba(X_test)
            if isinstance(clf, keras.wrappers.scikit_learn.KerasClassifier):
                pred_val = numpy.delete(pred_val, 0, axis=1)
                pred_test = numpy.delete(pred_test, 0, axis=1)

            blend_val[:, idx:idx + cls_cnt] += pred_val
            blend_test[:, idx:idx + cls_cnt] += pred_test

        blend_val[:, idx:idx + cls_cnt] /= clfs_cnt
        blend_test[:, idx:idx + cls_cnt] /= clfs_cnt

    blend_clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=util.seed)
    blend_clf.fit(blend_train, y_train)

    val_acc = blend_clf.score(blend_val, y_val)
    print('val_acc:', val_acc)

    pred = blend_clf.predict(blend_test)
    return val_acc, pred


def run(n_folds=5):
    """
    :param int n_folds:
    """
    print("Stacking Ensemble (Blend) using ExtraTrees with 5 models")
    util.init_random()

    X_test, test_id = feature.bow.build_test_set()

    acc_age, pred_age = build_blend_and_pred('age', X_test, n_folds)
    acc_gender, pred_gender = build_blend_and_pred('gender', X_test, n_folds)
    acc_education, pred_education = build_blend_and_pred('education', X_test, n_folds)

    acc_final = (acc_age + acc_gender + acc_education) / 3
    print('acc_final:', acc_final)

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education, '{file_name}.csv'.format(file_name=_file_name))
