#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import datasets


def main():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print('データセットの点数: {N}'.format(N=X.shape[0]))
    print('各データの次元数: {dimension}'.format(dimension=X.shape[1]))

    # データの中から 25 点を無作為に選び出す
    p = np.random.random_integers(0, len(X), 25)

    # 選んだデータとラベルを matplotlib で表示する
    samples = np.array(list(zip(X, y)))[p]
    for index, (data, label) in enumerate(samples):
        # 画像データを 5x5 の格子状に配置する
        plt.subplot(5, 5, index + 1)
        # 軸に関する表示はいらない
        plt.axis('off')
        # データを 8x8 のグレースケール画像として表示する
        plt.imshow(data.reshape(8, 8), cmap=cm.gray_r, interpolation='nearest')
        # 画像データのタイトルに正解ラベルを表示する
        plt.title(label, color='red')

    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()