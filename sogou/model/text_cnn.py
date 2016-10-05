# -*- coding: utf-8 -*-
"""
References:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
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

import feature
import submissions
import util


def build_clf(input_dim, output_dim, word_vec_dim=300, weights=None, img_name=None):
    """
    构建神经网络
    :param input_dim: 输入维数
    :param output_dim: 输出维数
    :param word_vec_dim: 词向量维数
    :param weights: 词向量权重矩阵
    :param img_name: 图片名称
    :rtype: keras.models.Model
    """
    if weights is not None:
        assert (feature.wv.word_counts + 1, word_vec_dim) == weights.shape
        weights = [weights]

    input_tensor = keras.layers.Input(shape=(input_dim,), dtype='int32')

    embedded = keras.layers.Embedding(input_dim=feature.wv.word_counts + 1,
                                      output_dim=word_vec_dim, input_length=input_dim,
                                      weights=weights)(input_tensor)

    tensors = []
    for filter_length in (3, 4, 5):
        tensor = keras.layers.Convolution1D(nb_filter=100, filter_length=filter_length, activation='relu')(embedded)
        tensor = keras.layers.MaxPooling1D(pool_length=input_dim - filter_length + 1)(tensor)
        tensor = keras.layers.Flatten()(tensor)
        tensors.append(tensor)

    output_tensor = keras.layers.merge(tensors, mode='concat', concat_axis=1)
    output_tensor = keras.layers.Dropout(0.5)(output_tensor)
    output_tensor = keras.layers.Dense(output_dim, activation='softmax')(output_tensor)

    clf = keras.models.Model(input_tensor, output_tensor)
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(clf.summary())

    if img_name:
        if not os.path.exists('img'):
            os.mkdir('img')
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val = feature.wv.build_train_set(label, validation_split=0.1, dummy=True)

    weights = feature.wv.build_weights_matrix(word_vec_dim=300)

    clf = build_clf(X_train.shape[1], y_train.shape[1], weights=weights,
                    img_name='img/{file_name}_{label}.png'.format(file_name=os.path.basename(__file__)[:-3],
                                                                  label=label))
    history = clf.fit(X_train, y_train, batch_size=128, nb_epoch=7, validation_data=(X_val, y_val), shuffle=True)

    val_acc = history.history['val_acc'][-1]
    print('val_acc:', val_acc)

    return clf, val_acc


def run():
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

    submissions.save_csv(test_id, pred_age, pred_gender, pred_education,
                         '{file_name}.csv'.format(file_name=os.path.basename(__file__)[:-3]))
