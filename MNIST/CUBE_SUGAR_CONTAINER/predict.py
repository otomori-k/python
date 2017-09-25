#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from sklearn import datasets
from sklearn import cross_validation
from sklearn import svm
from sklearn import metrics


def main():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    scores = []
    # K-fold 交差検証でアルゴリズムの汎化性能を調べる
    kfold = cross_validation.KFold(len(X), n_folds=10)
    for train, test in kfold:
        # デフォルトのカーネルは rbf になっている
        clf = svm.SVC(C=2**2, gamma=2**-11)
        # 訓練データで学習する
        clf.fit(X[train], y[train])
        # テストデータの正答率を調べる
        score = metrics.accuracy_score(clf.predict(X[test]), y[test])
        scores.append(score)

    # 最終的な正答率を出す
    accuracy = (sum(scores) / len(scores)) * 100
    msg = '正答率: {accuracy:.2f}%'.format(accuracy=accuracy)
    print(msg)


if __name__ == '__main__':
    main()