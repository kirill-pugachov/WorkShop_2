# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:23:38 2017

@author: Kirill
"""


# необходимые пакеты
import matplotlib.pyplot as plt
import numpy as np

#задаем размер изображения
plt.rcParams["figure.figsize"] = [18,12]

# фиксируем случайные величины
np.random.seed(42)

# генерируем диапазон зелёных точек
x1x2_green = np.random.randn(300, 2) * 4 + 20

# генерируем диапазон красных точек
x1x2_red = np.random.randn(300, 2) * 5 + 1

# все яйца в одну корзину
x1x2 = np.concatenate((x1x2_green, x1x2_red))

# проставляем классы: зелёные +1, красные -1
labels = np.concatenate((np.ones(x1x2_green.shape[0]), -np.ones(x1x2_red.shape[0])))

# перемешиваем
indices = np.array(range(x1x2.shape[0]))
np.random.shuffle(indices)
x1x2 = x1x2[indices]
labels = labels[indices]

# случайные начальные веса
w1_ = 2
w2_ = -2
b_ = 0

# разделяющая гиперплоскость (граница решений)
def lr_line(x1, x2):
    return w1_ * x1 + w2_ * x2 + b_

# ниже границы -1
# выше +1
def decision_unit(value):
    return -1 if value < 0 else 1

# добавляем начальное разбиение в список
lines = [[w1_, w2_, b_]]

for max_iter in range(300):
    # счётчик неверно классифицированных примеров
    # для ранней остановки
    mismatch_count = 0
    
    # по всем образцам
    for i, (x1, x2) in enumerate(x1x2):
        # считаем значение линейной комбинации на гиперплоскости
        value = lr_line(x1, x2)
        
        # класс из тренировочного набора (-1, +1)
        true_label = int(labels[i])
        
        # предсказанный класс (-1, +1)
        pred_label = decision_unit(value)
        
        # если имеет место ошибка классификации
        if (true_label != pred_label):
            # корректируем веса в сторону верного класса, т.е.
            # идём по нормали — (x1, x2) — в случае класса +1
            # или против нормали — (-x1, -x2) — в случае класса -1
            # т.к. нормаль всегда указывает в сторону +1
            w1_ = w1_ + x1 * true_label
            w2_ = w2_ + x2 * true_label
            
            # смещение корректируется по схожему принципу
            b_ = b_ + true_label
            
            # считаем количество неверно классифицированных примеров
            mismatch_count = mismatch_count + 1
    
    # если была хотя бы одна коррекция
    if (mismatch_count > 0):
        # запоминаем границу решений
        lines.append([w1_, w2_, b_])
    else:
        # иначе — ранняя остановка
        break

# рисуем точки (по последней границе решений)
for i, (x1, x2) in enumerate(x1x2):
    pred_label = decision_unit(lr_line(x1, x2))

    if (pred_label < 0):
        plt.plot(x1, x2, 'ro', color='red')
    else:
        plt.plot(x1, x2, 'ro', color='green')

# выставляем равное пиксельное разрешение по осям
plt.gca().set_aspect('equal', adjustable='box')    

# проставляем названия осей
plt.xlabel('x1')
plt.ylabel('x2')

# служебный диапазон для визуализации границы решений
x1_range = np.arange(-15, 30, 0.1)

# функционал, возвращающий границу решений в пригодном для отрисовки виде
# x2 = f(x1) = -(w1 * x1 + b) / w2
def f_lr_line(w1, w2, b):
    def lr_line(x1):
        return -(w1 * x1 + b) / w2
    
    return lr_line

# отрисовываем историю изменения границы решений
# если понадобилось много итераций выводим последние 20
it = 0
for coeff in lines:
    if len(lines) <= 20:
        lr_line = f_lr_line(coeff[0], coeff[1], coeff[2])        
        plt.plot(x1_range, lr_line(x1_range), label = 'it: ' + str(it))
    else:
        if it < (len(lines) - 20):
            lr_line = f_lr_line(coeff[0], coeff[1], coeff[2])
            plt.plot(x1_range, lr_line(x1_range))
        else:
            lr_line = f_lr_line(coeff[0], coeff[1], coeff[2])
            plt.plot(x1_range, lr_line(x1_range), label = 'it: ' + str(it)) 
    it = it + 1
    
# зум
plt.axis([-15, 30, -15, 30])
    
# легенда
plt.legend(loc = 'lower left')
  
# на экран!
plt.show()