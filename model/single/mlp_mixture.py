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

import conf
import feature.bow
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]
param = {'sparse': True, 'batch_size': 512, 'nb_epoch': 30}


def build_clf(input_dim, output_dim, summary=True, img_name=None):
    """
    构建神经网络
    :param int input_dim: 输入维数
    :param list output_dim: 输出维数
    :param bool summary: 是否输出网络结构
    :param str|unicode img_name: 图片名称
    :rtype: keras.models.Sequential
    """
    input_tensor = keras.layers.Input(shape=(input_dim,),
                                      sparse=param['sparse'])
    tensor = keras.layers.Dense(200, activation='relu')(input_tensor)
    tensor = keras.layers.Dropout(0.25)(tensor)
    output_tensors = []
    for dim in output_dim:
        output_tensor = keras.layers.Dense(dim, activation='softmax')(tensor)
        output_tensors.append(output_tensor)
    clf = keras.models.Model(input_tensor, output_tensors)
    clf.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])

    if summary:
        clf.summary()
    if img_name:
        if not os.path.exists(conf.IMG_DIR):
            os.mkdir(conf.IMG_DIR)
        keras.utils.visualize_util.plot(clf, to_file=img_name, show_shapes=True)
    return clf


def build():
    """
    构建分类器
    """
    X_train, y_train, X_val, y_val, y_shape = feature.bow_mixture.build_train(
        validation_split=0.1, sparse=param['sparse'])
    y_train_split = []
    y_val_split = []
    for shape in y_shape:
        y_train_split.append(y_train[:, :shape])
        y_train = y_train[:, shape:]
        y_val_split.append(y_val[:, :shape])
        y_val = y_val[:, shape:]
    best_model_path = '{temp}/{file}_best.hdf'.format(temp=conf.TEMP_DIR,
                                                      file=_file_name)

    clf = build_clf(X_train.shape[1], y_shape,
                    img_name='{img}/{file}.png'.format(img=conf.IMG_DIR,
                                                       file=_file_name))
    checkpoint = keras.callbacks.ModelCheckpoint(best_model_path,
                                                 monitor='val_acc', verbose=1,
                                                 save_best_only=True)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5,
                                              verbose=1)
    clf.fit(X_train, y_train_split, batch_size=param['batch_size'],
            nb_epoch=param['nb_epoch'], validation_data=(X_val, y_val_split),
            shuffle=True, callbacks=[checkpoint, earlystop])

    clf.load_weights(best_model_path)
    _, _, _, _, acc_age, acc_gender, acc_education = clf.evaluate(X_val,
                                                                  y_val_split)
    util.logger.info('acc_age: {acc}'.format(acc=acc_age))
    util.logger.info('acc_gender: {acc}'.format(acc=acc_gender))
    util.logger.info('acc_education: {acc}'.format(acc=acc_education))
    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    return clf


def run():
    util.logger.info('Multi-Layer Perceptron')
    util.init_random()

    clf = build()

    X_test = feature.bow.build_test(sparse=param['sparse'])

    pred_age, pred_gender, pred_education = clf.predict(X_test)
    pred_age = pred_age.argmax(axis=-1).flatten()
    pred_gender = pred_gender.argmax(axis=-1).flatten()
    pred_education = pred_education.argmax(axis=-1).flatten()

    submissions.save_csv(pred_age, pred_gender, pred_education,
                         '{file}.csv'.format(file=_file_name))
