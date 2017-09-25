import numpy as np
import pandas as pd
from pandas import Series,DataFrame

import math

#プロット用です
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline

# 機械学習用です。
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# もう一つ、性能の評価用に
from sklearn import metrics

# エラーが出たら、セットアップをお願いします。
import statsmodels.api as sm

# ロジスティック関数
def logistic(t):
    return 1.0 / (1 + math.exp((-1.0)*t) )

# tを-6 から 6まで 500 点用意します。
t = np.linspace(-6,6,500)

# リスト内包表記で、yを用意します。
y = np.array([logistic(ele) for ele in t])

# プロットしてみましょう。
plt.plot(t,y)
plt.title(' Logistic Function ')

# plt.show()

df = sm.datasets.fair.load_pandas().data

# 関数を作ります。
def affair_check(x):
    if x != 0:
        return 1
    else:
        return 0

# applyを使って、新しい列用のデータを作りましょう。
df['Had_Affair'] = df['affairs'].apply(affair_check)

# 不倫の有無（Had_Affair列）でグループ分けします。
df.groupby('Had_Affair').mean()

# 年齢分布を見てみます。
sns.countplot('age',data=df.sort('age'),hue='Had_Affair',palette='coolwarm')

# 年齢が上がると不倫率が上がる傾向が見えます。では、結婚してからの年月はどうでしょうか？
sns.countplot('yrs_married',data=df.sort('yrs_married'),hue='Had_Affair',palette='coolwarm')

# やはり結婚から年月が経つと、不倫率が上がるようです。
# 子供の数はどうでしょうか？
sns.countplot('children',data=df.sort('children'),hue='Had_Affair',palette='coolwarm')

# 子供の数が少ないと、不倫率が低いはっきりとした傾向があります。
# 最後は学歴を見ておきましょう。
sns.countplot('educ',data=df.sort('educ'),hue='Had_Affair',palette='coolwarm')

# 数字が大きいほど高学歴ですが、あまり関係が無いように見えます。他の列についてやってみるのもいいかもしれません。
# 引き続き、回帰モデルを作ることを目指します。

# カテゴリーを表現する変数を、ダミー変数に展開します。
occ_dummies = pd.get_dummies(df['occupation'])
hus_occ_dummies = pd.get_dummies(df['occupation_husb'])

occ_dummies.columns = ['occ1','occ2','occ3','occ4','occ5','occ6']
hus_occ_dummies.columns = ['hocc1','hocc2','hocc3','hocc4','hocc5','hocc6']

# 不要になったoccupationの列と、目的変数「Had_Affair」を削除します。
X = df.drop(['occupation','occupation_husb','Had_Affair'],axis=1)

# ダミー変数のDataFrameを繋げます。
dummies = pd.concat([occ_dummies,hus_occ_dummies],axis=1)

# 説明変数XのDataFrameです。
X = pd.concat([X,dummies],axis=1)

# Yに目的変数を格納します。
Y = df.Had_Affair

X = X.drop('occ1',axis=1)
X = X.drop('hocc1',axis=1)

X = X.drop('affairs',axis=1)

Y = Y.values
# または、
# Y = np.ravel(Y)

# LogisticRegressionクラスのインスタンスを作ります。
log_model = LogisticRegression() # fit_intercept=False, C=1e9) statsmodelsの結果に似せるためのパラメータ。

# データを使って、モデルを作ります。
log_model.fit(X,Y)

# モデルの精度を確認してみましょう。
log_model.score(X,Y)

# print(log_model.score(X,Y))

# 変数名とその係数を格納するDataFrameを作ります。
coeff_df = DataFrame([X.columns, log_model.coef_[0]]).T

# print(coeff_df)

# おなじく、train_test_splitを使います。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# 新しいモデルを作ります。
log_model2 = LogisticRegression()

# 学習用のデータだけでモデルを鍛えます。
log_model2.fit(X_train, Y_train)

# テスト用データを使って、予測してみましょう。
class_predict = log_model2.predict(X_test)

# 精度を計算してみます。
print(metrics.accuracy_score(Y_test,class_predict))

