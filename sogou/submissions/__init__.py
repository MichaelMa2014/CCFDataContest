# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import csv

import numpy
import pandas

import conf
import data


def save_csv(pred_age, pred_gender, pred_education, file_name):
    """
    :param list|numpy.ndarray pred_age: age预测值
    :param list|numpy.ndarray pred_gender: gender预测值
    :param list|numpy.ndarray pred_education: education预测值
    :param str|unicode file_name: 文件名
    """
    pandas.DataFrame(
        {'id': data.get_test_id(), 'age': pred_age, 'gender': pred_gender, 'education': pred_education}).to_csv(
        'submissions/%s' % file_name, sep=b' ', columns=['id', 'age', 'gender', 'education'], header=False, index=False,
        encoding=conf.ENCODING, quoting=csv.QUOTE_NONE)


def save_csv_by_df(pred_df, file_name):
    """
    :param pandas.DataFrame pred_df: 预测值
    :param str|unicode file_name: 文件名
    """
    pred_df.to_csv('submissions/%s' % file_name, sep=b' ', columns=['id', 'age', 'gender', 'education'], header=False,
                   index=False, encoding=conf.ENCODING, quoting=csv.QUOTE_NONE)
