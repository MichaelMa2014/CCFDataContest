# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import model.single.bnb
import model.single.et
import model.single.lr
import model.single.mnb
import model.single.rf
import model.single.svm
try:
    import model.single.tg
except ImportError:
    pass
import model.single.xgb

import model.single.cnn
import model.single.c_lstm
import model.single.fast_text
import model.single.mlp
import model.single.mlp_sklearn
