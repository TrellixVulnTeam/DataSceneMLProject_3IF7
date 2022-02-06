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


# Значение ключа DESCR – это краткое описание набора данных
print(iris_dataset['DESCR'][:193] + "\n...")

# Значение ключа target_names – это массив строк, содержащий сорта цветов
print("Названия ответов: {}".format(iris_dataset['target_names']))

# Значение  feature_names  –  это  список  строк  с  описанием  каждого признака:
print("Названия признаков: \n{}".format(iris_dataset['feature_names']))


# Сами  данные  записаны  в  массивах  target  и  data.  data  –  массив
# NumPy,  который  содержит  количественные  измерения  длины
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
# Значения чисел задаются массивом iris['target_names']: 0 – setosa,
# 1 – versicolor, а 2 – virginica .
print("Ответы:\n {}".format(iris_dataset['target']))




"""
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)


print("форма массива X_train: {}".format(X_train.shape))
print("форма массива y_train: {}".format(y_train.shape))

print("форма массива X_train: {}".format(X_test.shape))
print("форма массива y_train: {}".format(y_test.shape))

# создаем dataframe из данных в массиве X_train
# маркируем столбцы, используя строки в iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# создаем матрицу рассеяния из dataframe, цвет точек задаем с помощью y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

plt.show()


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

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Правильность на тестовом наборе: {:.2f}".format(knn.score(X_test, y_test)))


"""