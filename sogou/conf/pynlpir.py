# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import pynlpir


def init():
    """
    PyNLPIR分词初始化
    """
    # 设置PyNLPIR分词log级别
    pynlpir.logger.setLevel('INFO')
    # 设置PyNLPIR分词字典文件
    path = os.path.abspath('../util/pynlpir_dict').encode()
    pynlpir.open(path)
