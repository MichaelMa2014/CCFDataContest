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

import conf
import feature.wv
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
param = {'sparse': False, 'batch_size': 128, 'age': 40, 'gender': 40,
         'education': 40}


def build_clf(input_dim, output_dim, max_feature, word_vec_dim=100,
              summary=True, img_name=None):
    """
    构建神经网络
    :param int input_dim: 输入维数
    :param int output_dim: 输出维数
    :param int max_feature: 最大特征值
    :param int word_vec_dim: 词向量维数
    :param bool summary: 是否输出网络结构
    :param str|unicode img_name: 图片名称
    :rtype: keras.models.Sequential
    """
    input_tensor = keras.layers.Input(shape=(input_dim,),
                                      sparse=param['sparse'])
    tensor = keras.layers.Embedding(input_dim=max_feature + 1,
                                    output_dim=word_vec_dim,
                                    input_length=input_dim)(input_tensor)
    tensor = keras.layers.GlobalAveragePooling1D()(tensor)
    output_tensor = keras.layers.Dense(output_dim, activation='softmax')(tensor)
    clf = keras.models.Model(input_tensor, output_tensor)
    clf.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

    if summary:
        clf.summary()
    if img_name:
        if not os.path.exists(conf.IMG_DIR):
            os.mkdir(conf.IMG_DIR)
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def build(label):
    """
    构建分类器
    :param str|unicode label: 类别标签
    """
    X_train, y_train, X_val, y_val, max_feature = feature.wv.build_train(
        label, validation_split=0.1, dummy=True, sparse=param['sparse'])
    dir_path = '{temp}/{file}'.format(temp=conf.TEMP_DIR, file=_file_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    best_model_path = '{dir}/{label}_best.hdf'.format(dir=dir_path, label=label)

    clf = build_clf(X_train.shape[1], y_train.shape[1], max_feature,
                    img_name='{img}/{file}_{label}.png'.format(img=conf.IMG_DIR,
                                                               file=_file_name,
                                                               label=label))
    checkpoint = keras.callbacks.ModelCheckpoint(best_model_path,
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5,
                                              verbose=1)
    clf.fit(X_train, y_train, batch_size=param['batch_size'],
            nb_epoch=param[label], validation_data=(X_val, y_val), shuffle=True,
            callbacks=[checkpoint, earlystop])

    clf.load_weights(best_model_path)
    _, val_acc = clf.evaluate(X_val, y_val)
    util.logger.info('val_acc: {acc}'.format(acc=val_acc))

    return clf, val_acc


def run():
    util.logger.info('Fast Text')
    util.init_random()

    clf_age, acc_age = build('age')
    clf_gender, acc_gender = build('gender')
    clf_education, acc_education = build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    X_test = feature.wv.build_test(sparse=param['sparse'])

    pred_age = clf_age.predict(X_test).argmax(axis=-1).flatten()
    pred_gender = clf_gender.predict(X_test).argmax(axis=-1).flatten()
    pred_education = clf_education.predict(X_test).argmax(axis=-1).flatten()

    submissions.save_csv(pred_age, pred_gender, pred_education,
                         '{file}.csv'.format(file=_file_name))
