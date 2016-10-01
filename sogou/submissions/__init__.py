# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import csv

import pandas


def save_csv(ids, pred_age, pred_gender, pred_education, file_name):
    """
    :param list ids: 测试集id
    :param list pred_age: age预测值
    :param list pred_gender: gender预测值
    :param list pred_education: education预测值
    :param str|unicode file_name: 文件名
    """
    pandas.DataFrame({'id': ids, 'age': pred_age, 'gender': pred_gender, 'education': pred_education}).to_csv(
        './submissions/%s' % file_name, sep=' ', columns=['id', 'age', 'gender', 'education'], header=False,
        index=False, encoding='gb18030', quoting=csv.QUOTE_NONE)
