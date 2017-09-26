# 四則演算
1 + 2

1 - 2

4 * 5

7 / 5

# べき乗
3 ** 2

# データ型
type(10)

type(2.718)

type("hello")

# 変数
x = 10
print(x)

x = 100
print(x)

y = 3.14
x * y

type(x * y)

# リスト
a = [1, 2, 3, 4, 5]
print(a)

len(a)

a[0]
a[4]

a[4] = 99
print(a)

# スライシング
# 0番目から2番目まで（2番目は含まない）
a[0:2]

# 1番目から最後まで
a[1:]

# 最初から3番目まで（3番目は含まない）
a[:3]

# 最初から最後の要素の1つ前まで
a[:-1]

# 最初から最後の要素の2つ前まで
a[:-2]

# ディクショナリ
me = {'height':180}
me['height']

me['weight'] = 70
print(me)

# ブーリアン
hungry = True
sleepy = False
type(hungry)

not hungry

hungry and sleepy

hungry or sleepy

# if文
hungry = True
if hungry:
    print("I'm hungry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")

# for文
for i in [1, 2, 3]:
    print(i)

# 関数
def hello():
    print("Hello World!")

hello()

def hello(object):
    print("Hello " + object + "!")

hello("cat")
