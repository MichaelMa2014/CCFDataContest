# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import conf
import model

if __name__ == '__main__':
    conf.pynlpir.init()

    # model.mnb.run()
    # model.rf.run()
    # model.svm.run()

    model.cnn.run()
