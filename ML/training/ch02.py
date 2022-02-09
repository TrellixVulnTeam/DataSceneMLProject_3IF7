import numpy as np
import matplotlib.pyplot as plt
import mglearn

from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# генерируем набор данных
X, y = mglearn.datasets.make_forge()

# строим график для набора данных
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
# plt.show()

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")

#plt.show()


# Загружаем данные из scikit-learn с помощью  функции load_breast_cancer:
cancer = load_breast_cancer()
print("Ключи cancer(): \n{}".format(cancer.keys()))

"""
dict_keys(['feature_names', 'data', 'DESCR', 'target', 'target_names']) 

    feature_names– это список строк с описанием  каждого признака
    data- массив Nympu
    DESCR– это краткое описание набора данных
    target_names- массив строк
    target- Сами данные записаны в массивах
   
"""
print("Форма массива data для набора cancer: {}".format(cancer.data.shape))
print("Количество примеров для каждого класса:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))

# Чтобы получить содержательное описание каждого  признака, взглянем на атрибут feature_names:
print("Имена признаков:\n{}".format(cancer.feature_names))

"""
Out: 
Имена признаков: 
['mean radius' 'mean texture' 'mean perimeter' 'mean area' 
 'mean smoothness' 'mean compactness' 'mean concavity' 
 'mean concave points' 'mean symmetry' 'mean fractal dimension' 
 'radius error' 'texture error' 'perimeter error' 'area error' 
 'smoothness error' 'compactness error' 'concavity error' 
 'concave points error' 'symmetry error' 'fractal dimension error' 
 'worst radius' 'worst texture' 'worst perimeter' 'worst area' 
 'worst smoothness' 'worst compactness' 'worst concavity' 
 'worst concave points' 'worst symmetry' 'worst fractal dimension'] 
 
 """
# более  подробную  информацию о данных можно получить, прочитав cancer.DESCR.
print("подробную  информацию:\n{}".format(cancer.DESCR))

# Загружаем данные из scikit-learn с помощью  функции load_breast_cancer:
boston = load_boston()
print("форма массива data для набора boston: {}".format(boston.data.shape))

# более подробную информацию о данных можно получить, прочитав boston.DESCR
print("подробную  информацию:\n{}".format(boston.DESCR))

#  Набор данных c производными признаками можно загрузить с помощью функции load_extended_boston:
X, y = mglearn.datasets.load_extended_boston()
print("форма массива X: {}".format(X.shape))
print("форма массива y: {}".format(y.shape))

# Алгоритм k ближайших  соседей
# используются три ближайших соседа (n_neighbors=3):
mglearn.plots.plot_knn_classification(n_neighbors=3)


# мы разделим наши  данные на обучающий и тестовый наборы, чтобы  оценить обобщающую способность модели
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)
# Затем  подгоняем  классификатор,  используя  обучающий  набор
clf.fit(X_train, y_train)

"""
Чтобы получить прогнозы для тестовых данных, мы вызываем метод predict(). 
Для каждой точки тестового набора он вычисляет ее ближайших соседей  в  обучающем  наборе
и находит среди них наиболее часто встречающийся класс:
"""
print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))

"""
Для оценки обобщающей способности модели мы вызываем метод score() с тестовыми данными и тестовыми метками: 
"""
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))

# Мы видим, что наша модель имеет правильность 86%,
# то есть модель правильно предсказала класс для 86% примеров тестового набора



# АНАЛИЗ KNeighborsClassifier()

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # создаем объект-классификатор и подгоняем в одной строке
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("количество соседей:{}".format(n_neighbors))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
# plt.show()

print('*********************************\n...')
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
# пробуем n_neighbors от 1 до 10
neighbors_settings = range(1, 11)

# АНАЛИЗ KNeighborsClassifier

for n_neighbors in neighbors_settings:
    # строим модель
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # записываем правильность на обучающем наборе
    training_accuracy.append(clf.score(X_train, y_train))
    # записываем правильность на тестовом наборе
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()
