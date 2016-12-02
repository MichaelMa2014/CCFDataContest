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
import feature.ngram
import feature.wv
import model.single
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
validation_split = 0.1  # 验证集比例，如果为0.0则不返回验证集
n_folds = 5
skf = sklearn.model_selection.StratifiedKFold(n_folds, shuffle=True,
                                              random_state=util.seed)


def train_sub_clfs(X_train, y_train, X_val, X_test, clfs, **kwargs):
    """
    训练子分类器
    :param numpy.ndarray X_train:
    :param numpy.ndarray y_train:
    :param numpy.ndarray X_val:
    :param numpy.ndarray X_test:
    :param list clfs:
    """
    cls_cnt = len(numpy.unique(y_train))

    clfs_cnt = len(clfs)
    util.logger.info('classifiers count: {cnt}'.format(cnt=clfs_cnt))

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
                clf.load_weights(kwargs['base_clf_path'][clf_name])
                clf.fit(fold_X, fold_y, kwargs['param'][clf_name])
                clf.load_weights(kwargs['best_clf_path'][clf_name])

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
    return blend_train, blend_val, blend_test


def build_bow_clfs(label):
    """
    构建bow分类器
    :param str|unicode label: 类别标签
    """
    util.logger.info('classifiers using bag-of-words')
    X_train, y_train, X_val, y_val = feature.bow.build_train(
        label, validation_split=validation_split)
    X_test = feature.bow.build_test()

    # scikit-learn分类器
    clfs = [
        ('et', model.single.et.build_clf()),
        ('lr', model.single.lr.build_clf()),
        ('mlp_sklearn', model.single.mlp_sklearn.build_clf()),
        ('mnb', model.single.mnb.build_clf()),
        ('rf', model.single.rf.build_clf()),
        ('xgb', model.single.xgb.build_clf())
    ]

    # Keras神经网络
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
    clfs += [('mlp', model.single.mlp.build_clf(**param_build))]
    for clf_name, clf in [('mlp', model.single.mlp.build_clf(**param_build))]:
        clf.save_weights(base_clf_path[clf_name])
        clfs.append((clf_name, clf))

    blend_train, blend_val, blend_test = train_sub_clfs(
        X_train, y_train, X_val, X_test, clfs, param=param,
        base_clf_path=base_clf_path, best_clf_path=best_clf_path)
    return blend_train, y_train, blend_val, y_val, blend_test


def build_ngram_clfs(label, ngram=1):
    """
    构建ngram分类器
    :param str|unicode label: 类别标签
    :param int ngram:
    """
    util.logger.info('classifiers using {ngram}-ngram'.format(ngram=ngram))
    X_train, y_train, X_val, y_val, max_feature = feature.ngram.build_train(
        label, validation_split=validation_split, ngram=ngram)
    X_test = feature.ngram.build_test(ngram=ngram)

    param = {
        'fast_text': {'batch_size': model.single.fast_text.param['batch_size'],
                      'nb_epoch': model.single.fast_text.param[label]}}
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
                                            save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
        ]

    param_build = {'input_dim': X_train.shape[1],
                   'output_dim': len(numpy.unique(y_train)) + 1,
                   'max_feature': max_feature, 'summary': False}
    clfs = [('fast_text', model.single.fast_text.build_clf(**param_build))]
    for clf_name, clf in clfs:
        clf.save_weights(base_clf_path[clf_name])

    blend_train, blend_val, blend_test = train_sub_clfs(
        X_train, y_train, X_val, X_test, clfs, param=param,
        base_clf_path=base_clf_path, best_clf_path=best_clf_path)
    return blend_train, y_train, blend_val, y_val, blend_test


def build_wv_clfs(label):
    """
    构建wv分类器
    :param str|unicode label: 类别标签
    """
    util.logger.info('classifiers using word2vec')
    X_train, y_train, X_val, y_val, max_feature = feature.wv.build_train(
        label, validation_split=validation_split)
    X_test = feature.wv.build_test()

    param = {'cnn': {'batch_size': model.single.cnn.param['batch_size'],
                     'nb_epoch': model.single.cnn.param[label]}}
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
                                            save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
        ]

    param_build = {'input_dim': X_train.shape[1],
                   'output_dim': len(numpy.unique(y_train)) + 1,
                   'max_feature': max_feature, 'summary': False}
    clfs = [('cnn', model.single.cnn.build_clf(**param_build))]
    for clf_name, clf in clfs:
        clf.save_weights(base_clf_path[clf_name])

    blend_train, blend_val, blend_test = train_sub_clfs(
        X_train, y_train, X_val, X_test, clfs, param=param,
        base_clf_path=base_clf_path, best_clf_path=best_clf_path)
    return blend_train, y_train, blend_val, y_val, blend_test


def build_blend_and_pred(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    dir_path = '{temp}/{file}'.format(temp=conf.TEMP_DIR, file=_file_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    bow_data = build_bow_clfs(label)
    ngram_data = build_ngram_clfs(label, ngram=1)
    # wv_data = build_wv_clfs(label)

    for num in (0, 2, 4):
        assert bow_data[num].shape[0] == ngram_data[num].shape[
            0]  # == wv_data[num].shape[0]
    blend_train = numpy.zeros((bow_data[0].shape[0], 0))
    blend_val = numpy.zeros((bow_data[2].shape[0], 0))
    blend_test = numpy.zeros((bow_data[4].shape[0], 0))

    for num in (1, 3):
        assert bow_data[num].tolist() == ngram_data[
            num].tolist()  # == wv_data[num].tolist()
    y_train, y_val = bow_data[1], bow_data[3]

    # for sub_train, _, sub_val, _, sub_test in (bow_data, ngram_data, wv_data):
    for sub_train, _, sub_val, _, sub_test in (bow_data, ngram_data):
        blend_train = numpy.append(blend_train, sub_train, axis=1)
        blend_val = numpy.append(blend_val, sub_val, axis=1)
        blend_test = numpy.append(blend_test, sub_test, axis=1)

    blend_clf = sklearn.linear_model.LogisticRegression(n_jobs=-1,
                                                        random_state=util.seed)
    blend_clf.fit(blend_train, y_train)

    val_acc = blend_clf.score(blend_val, y_val)
    util.logger.info('val_acc: {acc}'.format(acc=val_acc))

    pred = blend_clf.predict(blend_test)
    return val_acc, pred


def run():
    util.logger.info('Stacking Ensemble (Blend) with 8 models')
    # util.logger.info('Stacking Ensemble (Blend) with 9 models')
    util.init_random()

    acc_age, pred_age = build_blend_and_pred('age')
    acc_gender, pred_gender = build_blend_and_pred('gender')
    acc_education, pred_education = build_blend_and_pred('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    submissions.save_csv(pred_age, pred_gender, pred_education,
                         '{file}.csv'.format(file=_file_name))
