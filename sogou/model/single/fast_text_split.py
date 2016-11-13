# -*- coding: utf-8 -*-
"""
References:
Bags of Tricks for Efficient Text Classification
https://arxiv.org/abs/1607.01759
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import keras
import keras.utils.visualize_util
import sklearn.metrics
import pandas

import feature.ngram_split
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
param = {'ngram': 1, 'batch_size': 128, 'age': 50, 'gender': 50, 'education': 50}


def build_clf(input_dim, output_dim, max_feature, word_vec_dim=100, img_name=None):
    """
    构建神经网络
    :param int input_dim: 输入维数
    :param int output_dim: 输出维数
    :param int max_feature: 最大特征值
    :param int word_vec_dim: 词向量维数
    :param str|unicode img_name: 图片名称
    :rtype: keras.models.Sequential
    """
    clf = keras.models.Sequential()
    clf.add(keras.layers.Embedding(input_dim=max_feature + 1, output_dim=word_vec_dim, input_length=input_dim))
    clf.add(keras.layers.GlobalAveragePooling1D())
    clf.add(keras.layers.Dense(output_dim, activation='softmax'))

    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    clf.summary()

    if img_name:
        if not os.path.exists('image'):
            os.mkdir('image')
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def merge(split_data, ids):
    """
    :param split_data:
    :param ids:
    :rtype: pandas.Series
    """
    df = pandas.DataFrame(split_data).join(pandas.DataFrame({'id': ids.tolist()}))
    return df.groupby('id').mean().idxmax(axis=1)


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val, ids_val, max_feature = feature.ngram_split.build_train_set(label,
                                                                                               validation_split=0.1,
                                                                                               dummy=True,
                                                                                               ngram=param['ngram'])
    best_model_path = 'temp/{file_name}_best.hdf'.format(file_name=_file_name)

    clf = build_clf(X_train.shape[1], y_train.shape[1], max_feature,
                    img_name='image/{file_name}_{label}_{ngram}gram.png'.format(file_name=_file_name, label=label,
                                                                                ngram=param['ngram']))
    checkpoint = keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_acc', save_best_only=True)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5)
    clf.fit(X_train, y_train, batch_size=param['batch_size'], nb_epoch=param[label], validation_data=(X_val, y_val),
            shuffle=True, callbacks=[checkpoint, earlystop])
    clf.load_weights(best_model_path)

    pred = clf.predict_proba(X_val)
    df = pandas.DataFrame({'y_true': pandas.Series(y_val, index=ids_val), 'y_pred': merge(pred, ids_val)})
    val_acc = sklearn.metrics.accuracy_score(df['y_true'], df['y_pred'])
    util.logger.info('val_acc: {acc}'.format(acc=val_acc))

    return clf, val_acc


def run():
    # TODO: 目前尚不能开启ngram>1，原因在于即使是ngram=2，max_feature也会爆炸到在8G内存下神经网络无法存放（Embedding层抛出MemoryError）
    assert param['ngram'] == 1

    util.logger.info('Fast Text')
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    X_test, ids = feature.ngram_split.build_test_set(ngram=param['ngram'])

    pred_age_split = clf_age.predict_proba(X_test)
    pred_age = merge(pred_age_split, ids)
    pred_gender_split = clf_gender.predict_proba(X_test)
    pred_gender = merge(pred_gender_split, ids)
    pred_education_split = clf_education.predict_proba(X_test)
    pred_education = merge(pred_education_split, ids)
    pred_df = pandas.DataFrame({'age': pred_age, 'gender': pred_gender, 'education': pred_education})

    submissions.save_csv_by_df(pred_df.reset_index(level='id'),
                               '{file_name}_{ngram}gram.csv'.format(file_name=_file_name, ngram=param['ngram']))
