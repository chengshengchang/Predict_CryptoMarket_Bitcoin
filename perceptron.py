from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

# 特徵矩陣
X = iris.data[:, [2, 3]]
# 目標
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

# 只需fit Train data
sc.fit(X_train)

# 要先tranform 標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

# eta 學習速率   n_iter 迭代次數

ppn = Perceptron(eta0=0.1, random_state=1,max_iter=40)

ppn.fit(X_train_std, y_train)

# 模型做預測
# y_pred 為預測類別標籤
y_pred = ppn.predict(X_test_std)

# y_test 為 真正類別標籤

# 準確率能用 accuracy_score 算 from sklearn
from sklearn.metrics import accuracy_score









