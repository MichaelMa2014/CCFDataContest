# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import generators
from __future__ import nested_scopes
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import sklearn.naive_bayes

import model.mnb.feature


def build(label):
    X_train, y_train, X_validation, y_validation = model.mnb.feature.get_train(label)

    clf = sklearn.naive_bayes.MultinomialNB()
    clf.fit(X_train, y_train)

    validation_result = clf.score(X_validation, y_validation)
    print(validation_result)
    return clf, validation_result


def run():
    clf_age, val_age = build('age')
    clf_gender, val_gender = build('gender')
    clf_education, val_education = build('education')

    val_final = (val_age + val_gender + val_education) / 3
    print(val_final)

    # X_test, test_id = model.mnb.feature.get_test();
    #
    # pred_age = clf_age.predict(X_test)
    #
    # submissions.save_csv(test_id, pred_age, '{file_name}.csv'.format(file_name=__file__[:-3]))
