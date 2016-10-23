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
    return pandas.read_csv('data/train.csv', names=['CONS_NO', 'LABEL'])


def load_test_user_data():
    """
    读入测试集数据
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/test_csv')


def load_kwh_data():
    """
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/temp/user_kwh.csv')


def load_pos_data():
    """
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/temp/pos.csv')


def load_neg_data():
    """
    :rtype: pandas.DataFrame
    """
    return pandas.read_csv('data/temp/neg.csv')


# 新建缓存
if not os.path.exists('data/temp'):
    os.mkdir('data/temp')

df = pandas.read_csv('data/all_user_yongdian_data_2015.csv')
df['DATA_DATE'] = pandas.to_datetime(df['DATA_DATE'])
pt = pandas.pivot_table(df, values='KWH', index=['CONS_NO'], columns=['DATA_DATE'])
ptT = pt.T
ptT = ptT.fillna(ptT.median())  # fillna 按列补值
pt = ptT.T
pt.to_csv('data/temp/user_kwh.csv', index=True, quoting=csv.QUOTE_NONE)
pt = pandas.read_csv('data/temp/user_kwh.csv')

# 划分数据集
train = load_train_user_data()
m = pandas.merge(pt, train, on='CONS_NO')
pos = m.loc[m.loc[:, 'LABEL'] == 1]
neg = m.loc[m.loc[:, 'LABEL'] == 0]
pos.to_csv('data/temp/pos.csv')
neg.to_csv('data/temp/neg.csv')