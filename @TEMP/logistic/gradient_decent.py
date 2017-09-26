# -*- coding:utf-8 -*-
import numpy as np

def gradient_decent(w, X, t, iter_num, alpha):
    M = float(len(X))
    J = []

    for i in range(iter_num):
        #現在のwから予測値を計算
        h = sigmoid(np.inner(w, X))
        #現在のwから目的関数を計算
        J.append(1/M * np.sum((-t*np.log(h) - (1-t)*np.log(1-h))))
        #説明変数が複数あることを想定し一つずつwを計算しリストに詰める
        w_set = []
        for k in range(len(w)):
            w_tmp = 1/M * np.sum((h - t) * X[:, k])
            w_set.append(w_tmp)
        #wの更新
        w = w - alpha*np.array(w_set)
    return w, J