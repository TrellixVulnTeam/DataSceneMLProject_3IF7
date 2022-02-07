import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Введение в машинное обучение с помощью Python [2017].pdf

# загрузить данных  Iris, вызвав функцию load_iris:
iris_dataset = load_iris()
print("Ключи iris_dataset: \n{}".format(iris_dataset.keys()))

"""
out:

Ключи iris_dataset: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

target_names - массив строк
feature_names - feature_names  –  это  список  строк  с  описанием  каждого признака
target - Сами  данные  записаны  в  массивах
data -массив Nympu
DESCR – это краткое описание набора данных
"""

# Значение ключа DESCR – это краткое описание набора данных
print(iris_dataset['DESCR'][:193] + "\n...")

# Значение ключа target_names – это массив строк, содержащий сорта цветов
print("Названия ответов: {}".format(iris_dataset['target_names']))

print("Форма массива: {}".format(iris_dataset['target_names'].shape))

print("Тип массива target_names: {}".format(type(iris_dataset['target_names'])))

# Значение  feature_names  –  это  список  строк  с  описанием  каждого признака:
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))


# Сами  данные  записаны  в  массивах  target  и  data.  
# data  –  массив  NumPy,  который  содержит  количественные  измерения  длины
# чашелистиков,  ширины  чашелистиков,  длины  лепестков  и  ширины  лепестков
print("Тип массива data: {}".format(type(iris_dataset['data'])))

# Строки  в  массиве  data  соответствуют  цветам  ириса,  а  столбцы
# представляют  собой  четыре  признака,  которые  были  измерены  для
# каждого цветка:
print("Форма массива data: {}".format(iris_dataset['data'].shape))

# признаками ( feature ). Форма (shape)  массива
# данных  определяется  количеством  примеров,  умноженным  на количество  признаков.
print("Первые пять строк массива data:\n{}".format(['data'][:5]))

# Массив  target  содержит  сорта  уже  измеренных  цветов,  тоже записанные в виде массива NumPy
print("Тип массива target: {}".format(iris_dataset['target']))


print("Форма массива target: {}".format(iris_dataset['target'].shape))

# Сорта кодируются как целые числа от 0 до 2:
# Значения чисел задаются массивом iris['target_names']: 0 – setosa, 1 – versicolor, а 2 – virginica .
print("Ответы:\n {}".format(iris_dataset['target']))

"""
out
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

0 – setosa, 1 – versicolor, а 2 – virginica.

"""


# Метрика эффективности

"""
X так обозначают данные
y так обозначают метки

train_test_split  для  наших  данных  и зададим  обучающие  данные,  обучающие  метки,  тестовые  данные, тестовые метки

функция  train_test_split  перемешивает  набор данных с помощью генератора псевдослучайных чисел random_state.


Выводом  функции  train_test_split  являются  X_train,  X_test,y_train  и  y_test,  которые  все  являются  массивами  Numpy.  
X_train содержит 75% строк набора данных, а X_test содержит оставшиеся 25%: 


"""


X_train, X_test, y_train, y_test = train_test_split( iris_dataset['data'], iris_dataset['target'], random_state=0)

# данные для тренировки
print("форма массива X_train: {}".format(X_train.shape))

# тренеровочные метки                        
print("форма массива y_train: {}".format(y_train.shape))

# данные для теста
print("форма массива X_test: {}".format(X_test.shape))

# тренировочные метки
print("форма массива y_test: {}".format(y_test.shape))


# Один из лучших способов исследовать данные – визуализировать их.
# Это  можно  сделать,  используя диаграмму  рассеяния (scatter  plot) 

"""

# создаем dataframe из данных в массиве X_train
# маркируем столбцы, используя строки в iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# В  pandas  есть  функция  для  создания парных  диаграмм  рассеяния  под  названием  scatter_matrix
# создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

plt.show()

"""

"""
Теперь мы можем начать строить реальную модель машинного обучения. 
В  библиотеке  scikit-learn  имеется  довольно  много  алгоритмов 
классификации,  которые  мы  могли  бы  использовать  для  построения 
модели.
В  scikit-learn  все  модели  машинного  обучения  реализованы  в 
собственных  классах,  называемых  классами  Estimator.

Алгоритм классификации  на  основе  метода k ближайших  соседей  реализован  в 
классификаторе  KNeighborsClassifier  модуля  neighbors.

"""

# создаem  объект-экземпляр класса
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


X_new = np.array([[5, 2.9, 1, 0.2]])
print("форма массива X_new: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Прогноз: {}:".format(prediction))
print("Спрогнозированная метка: {}".format(
       iris_dataset['target_names'][prediction]))


# Оценка качества моделей
y_pred = knn.predict(X_test)
print("Прогнозы для тестового набора:\n {}".format(y_pred))

print("Правильность на тестовом наборе: {:.2f}".format(np.mean(y_pred == y_test)))



print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))

""" эти строки повторяются  выше

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))


"""