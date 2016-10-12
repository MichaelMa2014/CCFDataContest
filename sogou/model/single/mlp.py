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

import feature.bow
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
param = {'batch_size': 512, 'age': 2, 'gender': 1, 'education': 2}


def build_clf(input_dim, output_dim, img_name=None):
    """
    构建神经网络
    :param int input_dim: 输入维数
    :param int output_dim: 输出维数
    :param str|unicode img_name: 图片名称
    :rtype: keras.models.Sequential
    """
    clf = keras.models.Sequential()
    clf.add(keras.layers.Dense(1024, activation='relu', input_dim=input_dim))
    clf.add(keras.layers.Dense(256, activation='relu'))
    clf.add(keras.layers.Dense(output_dim, activation='softmax'))

    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(clf.summary())

    if img_name:
        if not os.path.exists('image'):
            os.mkdir('image')
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.bow.build_train_set(label, validation_split=0.1, dummy=True)

    clf = build_clf(X_train.shape[1], y_train.shape[1],
                    img_name='image/{file_name}_{label}.png'.format(file_name=_file_name, label=label))
    history = clf.fit(X_train, y_train, batch_size=param['batch_size'], nb_epoch=param[label],
                      validation_data=(X_val, y_val), shuffle=True)

    val_acc = history.history['val_acc'][-1]
    print('val_acc:', val_acc)

    return clf, val_acc


def run():
    print("Multi-Layer Perceptron")
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

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education, '{file_name}.csv'.format(file_name=_file_name))
