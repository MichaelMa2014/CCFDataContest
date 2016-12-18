# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import os

import feature.wv
import model.single
import model.ensemble
import submissions
import util

_file_name = os.path.splitext(os.path.basename(__file__))[0]


def run():
    util.logger.info('Mixture')
    util.init_random()

    acc_age, pred_age = model.ensemble.stacking_all.build_blend_and_pred('age')
    clf_gender, acc_gender = model.single.fast_text.build('gender')
    clf_education, acc_education = model.single.fast_text.build('education')

    acc_final = (acc_age + acc_gender + acc_education) / 3
    util.logger.info('acc_final: {acc}'.format(acc=acc_final))

    X_test = feature.wv.build_test(
        sparse=model.single.fast_text.param['sparse'])
    pred_gender = clf_gender.predict(X_test).argmax(axis=-1).flatten()
    pred_education = clf_education.predict(X_test).argmax(axis=-1).flatten()

    submissions.save_csv(pred_age, pred_gender, pred_education,
                         '{file}.csv'.format(file=_file_name))
