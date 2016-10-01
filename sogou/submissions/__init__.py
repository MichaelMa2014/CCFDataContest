# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import pandas


def save_csv(ids, pred_age, pred_gender, pred_education, file_name):
    pandas.DataFrame({'id': ids, 'age': pred_age, 'gender': pred_gender, 'education': pred_education}).to_csv(
        './submissions/%s' % file_name, header=False, index=False, quoting=3)
