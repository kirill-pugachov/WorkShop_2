# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:50:51 2017

@author: Kirill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#загружаем данные
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 
'Alcalinity of ash', 'Magnesium', 'Total phenols', 
'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


#перемешиваем данные в сете случайным образом
df_wine = df_wine.iloc[np.random.permutation(len(df_wine))]

#делим данные на тестовую и обучающую выборки
tr = 0.7
ts = 0.3


train = df_wine[0:int(len(df_wine) * tr)]
test = df_wine[int(len(df_wine) * 0.7):]


y_train = np.array(train['Class label'])
X_train = np.array(train.drop('Class label', 1))


y_test = np.array(test['Class label'])
X_test = np.array(test.drop('Class label', 1))


#стандартизируем данные
X_train_std = (X_train-X_train.mean())/X_train.std()
X_test_std = (X_test-X_test.mean())/X_test.std()

#Расчитываем матрицу ковариации
cov_mat = np.cov(X_train_std.T)
#Получаем собственные значения и вектора
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Делаем список собственных чисел и векторов в таплах
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Сортируем таплы от большего к меньшему
eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W:\n', w)

X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], 
                X_train_pca[y_train==l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
