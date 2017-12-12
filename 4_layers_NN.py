# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:16:31 2017

@author: Kirill
"""

import numpy as np

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)
	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,0,1],
            [0,1,1,1],
            [1,0,1,0],
            [1,1,1,1],
            [1,1,1,0]])
                
y = np.array([[0,0],
			[1,1],
			[0,0],
			[1,1],
            [1,1]])

    
X_test = np.array([[0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0],
                  [0,0,0,0]])
np.random.seed(1)


#случайная инициализация весов синапсов со средним значением 0
syn0 = 2*np.random.random((4,4)) - 1
syn1 = 2*np.random.random((4,4)) - 1
syn2 = 2*np.random.random((4,4)) - 1
syn3 = 2*np.random.random((4,2)) - 1

for j in range(120000):

# Запускаем данные через слои 0, 1, 2, 3
    l0 = X
#    print('l0', l0.shape)
    l1 = nonlin(np.dot(l0,syn0))
#    print('l1', l1.shape)
    l2 = nonlin(np.dot(l1,syn1))
#    print('l2', l2.shape)
    l3 = nonlin(np.dot(l2,syn2))
#    print('l3', l3.shape)
    l4 = nonlin(np.dot(l3,syn3))

# Рассчитываем ошибку на выходном слое и известным значением
    l4_error = y - l4
#    print('l3_error', l3_error.shape)

#Каждые 10000 итераций печатаем значение ошибки    
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l4_error))), '\n')
        
# Запускаем расчет ошибки на каждом слое
# и определяем величину изменения весов на синапсах
    l4_delta = l4_error*nonlin(l4, deriv=True)
#    print('l3_delta', l3_delta.shape)
    
    l3_error = l4_delta.dot(syn3.T)
    l3_delta = l3_error * nonlin(l3, deriv=True)

    l2_error = l3_delta.dot(syn2.T)
#    print('l2_error', l2_error.shape)
    l2_delta = l2_error * nonlin(l2, deriv=True)
#    print('l2_delta', l2_delta.shape)
    
    l1_error = l2_error.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)


#Пересчитываем значения весов на синапсах    
    syn3 += l3.T.dot(l4_delta)
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


print ("Output After Training NN:")
print(l4, '\n')


#Предсказываем на обученной NN по тестовому набору
l0 = X_test
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))
l3 = nonlin(np.dot(l2,syn2))
l4 = nonlin(np.dot(l3,syn3))
    

print ("Predictes Result on Educated NN :")
print(l4, '\n')