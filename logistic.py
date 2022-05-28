from sklearn.linear_model import LogisticRegression

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 特徵矩陣
X = iris.data[:, [2, 3]]
# 類別目標
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

'''
c = 權重參數 越小 正規化 強度越強


正規化降低加權 ， 讓 overfitting 降低
'''
lr  = LogisticRegression(C=100,random_state=1)

lr.fit(X_train,y_train)



# 預測類別機率

# 拿出看前三行的預測類別機率
lr.predict_proba(X_test_std[:3,:])
'''
>>>>
[[9.99977026e-01 2.29743060e-05 2.37163898e-18]  9.999代表 有這麼高的低率是第一類
 [1.00000000e+00 3.15791415e-11 7.70033170e-39]   7.7 是這類
 [1.00000000e+00 1.08084973e-11 2.96390892e-40]]
 
 所以可以通過 每航最大值去得到 預測的類別
 lr.predict_proba(X_test_std[:3,:]).argmax(axis=1)
 
 也可以直接用predict fuction
'''
# X_test[行,列]
lr.predict(X_test[:3,:])
'''
>>>[2 0 0]
'''
lr.predict_proba(X_test_std[:3,:]).argmax(axis=1)

# sklearn預測單一樣本需要 將2d array 當作數據
# 所以要將 1d --> 2d   python pakage == > reshape

print(lr.predict(X_test_std[0,:].reshape(1,-1)))
'''
預測第一行的結果

>>> [0]
'''


