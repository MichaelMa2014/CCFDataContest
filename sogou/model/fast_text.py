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

import feature.ngram
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def build_clf(input_dim, output_dim, max_feature, word_vec_dim=300, img_name=None):
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
    print(clf.summary())

    if img_name:
        if not os.path.exists('image'):
            os.mkdir('image')
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def build(label, ngram, nb_epoch):
    """
    构建分类器
    :param str|unicode label: 类别标签
    :param int ngram:
    :param int nb_epoch:
    """
    X_train, y_train, X_val, y_val, max_feature = feature.ngram.build_train_set(label, validation_split=0.1, dummy=True,
                                                                                ngram=ngram)

    clf = build_clf(X_train.shape[1], y_train.shape[1], max_feature,
                    img_name='image/{file_name}_{label}_ngram{ngram}.png'.format(file_name=_file_name, label=label,
                                                                                 ngram=ngram))
    history = clf.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epoch, validation_data=(X_val, y_val), shuffle=True)

    val_acc = history.history['val_acc'][-1]
    print('val_acc:', val_acc)

    return clf, val_acc


def run(ngram=1):
    """
    :param int ngram:
    """
    assert ngram == 1  # TODO: 目前尚不能开启ngram>1，原因在于即使是ngram=2，max_feature也会爆炸到在8G内存下神经网络无法存放（Embedding层抛出MemoryError）
    util.init_random()

    clf_age, acc_age = build('age', ngram=ngram, nb_epoch=9)
    clf_gender, acc_gender = build('gender', ngram=ngram, nb_epoch=6)
    clf_education, acc_education = build('education', ngram=ngram, nb_epoch=12)

    acc_final = (acc_age + acc_gender + acc_education) / 3
    print('acc_final:', acc_final)

    X_test, test_id = feature.ngram.build_test_set(ngram=ngram)

    pred_age = clf_age.predict_classes(X_test).flatten()
    pred_gender = clf_gender.predict_classes(X_test).flatten()
    pred_education = clf_education.predict_classes(X_test).flatten()

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education,
                         '{file_name}_ngram{ngram}.csv'.format(file_name=_file_name, ngram=ngram))
