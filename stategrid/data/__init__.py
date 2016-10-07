# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import csv
import os

import pandas


def load_train_user_data():
    """
    读入训练集数据
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('train.csv')


def load_test_user_data():
    """
    读入测试集数据
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('test_csv')


def load_kwh_data():
    """
    :rtype: pandas.DataFrame
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
    path = os.path.abspath('temp/user_kwh.csv')
    if not os.path.exists(path):
        df = pandas.read_csv('all_user_yongdian_data_2015.csv')
        df['DATA_DATE'] = pandas.to_datetime(df['DATA_DATE'])
        pt = pandas.pivot_table(df, values='KWH', index=['CONS_NO'], columns=['DATA_DATE'])
        pt.to_csv(path, index=True, quoting=csv.QUOTE_NONE)
    return pandas.read_csv(path)
