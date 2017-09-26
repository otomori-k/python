# SQLコマンドが使える
import pandas as pd
# 行列計算
import numpy as np
from pandas import DataFrame
# 機械学習のモデルや前処理
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# データの可視化
# %matplotlib inline

# import seaborn as sns

# ファイルを読み込む
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#利用しない変数は削除します
train_df = train_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)
test_df = test_df.drop(['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'],axis=1)

#年齢の欠損値を男女の平均年齢で補間
age_train_mean = train_df.groupby('Sex').Age.mean()
 
def fage(x):
    if x.Sex == 'male':
        return round(age_train_mean['male'])
    if x.Sex == 'female':
        return round(age_train_mean['female'])
 
train_df.Age.fillna(train_df[train_df.Age.isnull()].apply(fage,axis=1),inplace=True)

age_test_mean = test_df.groupby('Sex').Age.mean()
 
def fage(x):
    if x.Sex == 'male':
        return round(age_test_mean['male'])
    if x.Sex == 'female':
        return round(age_test_mean['female'])
 
test_df.Age.fillna(test_df[test_df.Age.isnull()].apply(fage,axis=1),inplace=True)

#クロス集計
sex_ct = pd.crosstab(train_df['Sex'], train_df['Survived'])
print(sex_ct)

#Femaleカラムを追加し、Sex要素のmale/femaleを1/0に変換して、要素として追加する
train_df['Female'] = train_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
test_df['Female'] = test_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
 
#クロス集計
pclass_ct = pd.crosstab(train_df['Pclass'], train_df['Survived'])
pclass_ct

#Pclassをダミー変数で分ける
pclass_train_df  = pd.get_dummies(train_df['Pclass'],prefix='Class')
pclass_test_df  = pd.get_dummies(test_df['Pclass'],prefix='Class')
 
#Class_3を削除
pclass_train_df = pclass_train_df.drop(['Class_3'], axis=1)
pclass_test_df = pclass_test_df.drop(['Class_3'], axis=1)
 
#Class_1,Class_2カラムを追加
train_df = train_df.join(pclass_train_df)
test_df = test_df.join(pclass_test_df)

X = train_df.drop(['PassengerId','Survived','Pclass','Sex'],axis=1)
y = train_df.Survived
 
#モデルの生成
clf = LogisticRegression()
 
#学習
clf.fit(X, y)

#学習したモデルの精度
clf.score(X,y)

print(clf.score(X,y))

#変数名とその係数を格納するデータフレーム
coeff_df = DataFrame([X.columns, clf.coef_[0]]).T
coeff_df

print(np.exp(2.46126))

#テストデータから生存者を予測
test_df = test_df.drop(['Sex','PassengerId','Pclass'],axis=1)
test_predict = clf.predict(test_df)
# print(test_predict)

# test_df = pd.read_csv('data/test.csv')
# result_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived':np.array(test_predict)})
