# -*- coding: utf-8 -*-

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

import feature
import submissions
import util


def build_clf(input_dim, output_dim):
    """
    构建神经网络
    :param input_dim: 输入维数
    :param output_dim: 输出维数
    :rtype: keras.models.Sequential
    """
    clf = keras.models.Sequential()
    clf.add(keras.layers.Dense(1024, activation='relu', input_dim=input_dim))
    clf.add(keras.layers.Dense(256, activation='relu'))
    clf.add(keras.layers.Dense(output_dim, activation='softmax'))

    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(clf.summary())

    if not os.path.exists('img'):
        os.mkdir('img')
    keras.utils.visualize_util.plot(clf, to_file='img/{file_name}.png'.format(file_name=__file__[:-3]),
                                    show_shapes=True)
    return clf


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, validation_split=0.1, dummy=True)

    clf = build_clf(X_train.shape[1], y_train.shape[1])
    clf.fit(X_train, y_train, batch_size=128, nb_epoch=20, validation_split=0.1, shuffle=True)

    val_loss, val_acc = clf.evaluate(X_val, y_val)
    print('val_loss: %f - val_acc: %f' % (val_loss, val_acc))

    return clf, val_acc


def run():
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    print('acc_final:', acc_final)

    X_test, test_id = feature.bow.build_test_set()

    pred_age = clf_age.predict_classes(X_test).flatten()
    pred_gender = clf_gender.predict_classes(X_test).flatten()
    pred_education = clf_education.predict_classes(X_test).flatten()

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education,
                         '{file_name}.csv'.format(file_name=__file__[:-3]))
