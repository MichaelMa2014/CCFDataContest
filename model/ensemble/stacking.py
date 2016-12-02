# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import keras.callbacks
import keras.utils.np_utils
import numpy
import sklearn.linear_model
import sklearn.model_selection

import conf
import feature.bow
import model.single
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
n_folds = 5


def build_blend_and_pred(label, X_test):
    """
    构建分类器
    :param str|unicode label: 类别标签
    :param numpy.ndarray X_test:
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train(
        label, validation_split=0.1)

    clfs = [
        ('et', model.single.et.build_clf()),
        ('lr', model.single.lr.build_clf()),
        ('mlp_sklearn', model.single.mlp_sklearn.build_clf()),
        ('mnb', model.single.mnb.build_clf()),
        ('rf', model.single.rf.build_clf()),
        ('xgb', model.single.xgb.build_clf())
    ]

    param = {'mlp': {'batch_size': model.single.mlp.param['batch_size'],
                     'nb_epoch': model.single.mlp.param[label]}}
    dir_path = '{temp}/{file}'.format(temp=conf.TEMP_DIR, file=_file_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    base_clf_path = {}
    best_clf_path = {}
    for c in param:
        base_clf_path[c] = '{dir}/{clf}_{label}_base.hdf'.format(dir=dir_path,
                                                                 clf=c,
                                                                 label=label)
        best_clf_path[c] = '{dir}/{clf}_{label}_best.hdf'.format(dir=dir_path,
                                                                 clf=c,
                                                                 label=label)
        param[c]['shuffle'] = True
        param[c]['validation_data'] = (
            X_val, keras.utils.np_utils.to_categorical(y_val))
        param[c]['callbacks'] = [
            keras.callbacks.ModelCheckpoint(best_clf_path[c], monitor='val_acc',
                                            verbose=1, save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_acc', patience=5,
                                          verbose=1)
        ]

    param_build = {'input_dim': X_train.shape[1],
                   'output_dim': len(numpy.unique(y_train)) + 1,
                   'summary': False}
    for clf_name, clf in [('mlp', model.single.mlp.build_clf(**param_build))]:
        clf.save_weights(base_clf_path[clf_name])
        clfs.append((clf_name, clf))

    cls_cnt = len(numpy.unique(y_train))
    util.logger.info('classes count: {cnt}'.format(cnt=cls_cnt))
    clfs_cnt = len(clfs)
    util.logger.info('classifiers count: {cnt}'.format(cnt=clfs_cnt))

    skf = sklearn.model_selection.StratifiedKFold(n_folds, shuffle=True,
                                                  random_state=util.seed)

    # TODO mabe need to be sparse
    blend_train = numpy.zeros((X_train.shape[0], clfs_cnt * cls_cnt))
    blend_val = numpy.zeros((X_val.shape[0], clfs_cnt * cls_cnt))
    blend_test = numpy.zeros((X_test.shape[0], clfs_cnt * cls_cnt))

    for i, (clf_name, clf) in enumerate(clfs):
        if isinstance(clf, keras.models.Model):
            util.logger.info('classifier No.{i}:'.format(i=i + 1))
            clf.summary()
        else:
            util.logger.info(
                'classifier No.{i}: {clf}'.format(i=i + 1, clf=clf))
        idx = i * cls_cnt

        for j, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
            util.logger.info('fold: {fold}'.format(fold=j + 1))

            fold_X, fold_y = X_train[train_idx], y_train[train_idx]
            fold_test = X_train[test_idx]

            if isinstance(clf, keras.models.Model):
                fold_y = keras.utils.np_utils.to_categorical(fold_y)
                clf.load_weights(base_clf_path[clf_name])
                clf.fit(fold_X, fold_y, **param[clf_name])
                clf.load_weights(best_clf_path[clf_name])

                fold_pred = clf.predict(fold_test)
                fold_pred = numpy.delete(fold_pred, 0, axis=1)
                pred_val = clf.predict(X_val)
                pred_val = numpy.delete(pred_val, 0, axis=1)
                pred_test = clf.predict(X_test)
                pred_test = numpy.delete(pred_test, 0, axis=1)
            else:
                clf.fit(fold_X, fold_y)

                fold_pred = clf.predict_proba(fold_test)
                pred_val = clf.predict_proba(X_val)
                pred_test = clf.predict_proba(X_test)

            blend_train[test_idx, idx:idx + cls_cnt] = fold_pred
            blend_val[:, idx:idx + cls_cnt] += pred_val
            blend_test[:, idx:idx + cls_cnt] += pred_test

        blend_val[:, idx:idx + cls_cnt] /= clfs_cnt
        blend_test[:, idx:idx + cls_cnt] /= clfs_cnt

    blend_clf = sklearn.linear_model.LogisticRegression(n_jobs=-1,
                                                        random_state=util.seed)
    blend_clf.fit(blend_train, y_train)

    val_acc = blend_clf.score(blend_val, y_val)
    util.logger.info('val_acc: {acc}'.format(acc=val_acc))

    pred = blend_clf.predict(blend_test)
    return val_acc, pred


def run():
    util.logger.info('Stacking Ensemble (Blend) with 7 models')
    util.init_random()

    X_test = feature.bow.build_test()

    acc_age, pred_age = build_blend_and_pred('age', X_test)
    acc_gender, pred_gender = build_blend_and_pred('gender', X_test)
    acc_education, pred_education = build_blend_and_pred('education', X_test)

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    submissions.save_csv(pred_age, pred_gender, pred_education,
                         '{file}.csv'.format(file=_file_name))
