# -*- coding: utf-8 -*-
"""
References:
Convolutional Neural Networks for Sentence Classification
https://arxiv.org/abs/1408.5882
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

import feature.wv
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
param = {'batch_size': 128, 'age': 5, 'gender': 4, 'education': 4}


def build_clf(input_dim, output_dim, max_feature, word_vec_dim=300, with_weights=True, img_name=None):
    """
    构建神经网络
    :param int input_dim: 输入维数
    :param int output_dim: 输出维数
    :param int max_feature: 最大特征值
    :param int word_vec_dim: 词向量维数
    :param bool with_weights: 初始化时是否引入词向量权重
    :param str|unicode img_name: 图片名称
    :rtype: keras.models.Model
    """
    weights = [feature.build_weights_matrix(word_vec_dim=300)] if with_weights else None

    input_tensor = keras.layers.Input(shape=(input_dim,), dtype='int32')
    embedded = keras.layers.Embedding(input_dim=max_feature + 1, output_dim=word_vec_dim, input_length=input_dim,
                                      weights=weights)(input_tensor)

    tensors = []
    for filter_length in (3, 4, 5):
        tensor = keras.layers.Convolution1D(nb_filter=100, filter_length=filter_length, activation='relu')(embedded)
        tensor = keras.layers.GlobalMaxPooling1D()(tensor)
        tensors.append(tensor)

    output_tensor = keras.layers.merge(tensors, mode='concat', concat_axis=1)
    output_tensor = keras.layers.Dropout(0.5)(output_tensor)
    output_tensor = keras.layers.Dense(output_dim, activation='softmax')(output_tensor)

    clf = keras.models.Model(input_tensor, output_tensor)
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
    X_train, y_train, X_val, y_val, max_feature = feature.wv.build_train_set(label, validation_split=0.1, dummy=True)

    clf = build_clf(X_train.shape[1], y_train.shape[1], max_feature,
                    img_name='image/{file_name}_{label}.png'.format(file_name=_file_name, label=label))
    history = clf.fit(X_train, y_train, batch_size=param['batch_size'], nb_epoch=param[label],
                      validation_data=(X_val, y_val), shuffle=True)

    val_acc = history.history['val_acc'][-1]
    print('val_acc:', val_acc)

    return clf, val_acc


def run():
    print("TextCNN")
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    print('acc_final:', acc_final)

    X_test, test_id = feature.wv.build_test_set()

    pred_age = clf_age.predict(X_test).argmax(axis=-1).flatten()
    pred_gender = clf_gender.predict(X_test).argmax(axis=-1).flatten()
    pred_education = clf_education.predict(X_test).argmax(axis=-1).flatten()

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education, '{file_name}.csv'.format(file_name=_file_name))
