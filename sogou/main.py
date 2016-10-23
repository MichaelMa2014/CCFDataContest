# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import conf
import model.ensemble
import model.single

if __name__ == '__main__':
    conf.pynlpir.init()

    # model.single.bnb.run()
    # model.single.et.run()
    # model.single.lr.run()
    # model.single.mnb.run()
    # model.single.rf.run()
    # model.single.svm.run()
    # model.single.tg.run()
    # model.single.xgb.run()

    # model.single.fast_text.run()
    model.single.cnn.run()
    # model.single.lstm.run()
    # model.single.mlp.run()
    # model.single.mlp_sklearn.run()
    # model.single.text_cnn.run()

    # model.ensemble.stacking.run()
    model.ensemble.stacking_all.run()
    # model.ensemble.voting_hard.run()
    # model.ensemble.voting_soft.run()
