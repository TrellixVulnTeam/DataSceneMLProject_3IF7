# Продвинутый уровень понимания линейной регрессии

"""
Давайте посмотрим, как обучить модель линейной регрессии, пользуясь только библиотечными функциями - имеенно их вы будете применять при решении реальных задач на работе

Сначала реализуем вспомогательную функцию для печати чисел питоновского типа float в красивом виде без большого количества знаков после запятой:
"""

from sklearn.datasets import load_boston
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

def ndprint(a, format_string ='{0:.2f}'):
    """Функция, которая распечатывает список в красивом виде"""
    return [format_string.format(v,i) for i,v in enumerate(a)]

"""
Загружаем исходные данные - датасет с ценами на дома в Бостоне. Это стандартный датасет, 
который используется для демонстрации алгоритмов настолько часто, что включён прямо в исходный код библиотеки sklearn.
"""


boston_dataset = load_boston()
features = boston_dataset.data
y = boston_dataset.target

print('Матрица Объекты X Фичи  (размерность): %s %s' % features.shape)
print('\nЦелевая переменная y (размерность): %s' % y.shape)


# текстовое описание датасета  - распечатать, если интересно print('\n',boston_dataset.DESCR


# вычисляем к-ты линейной регрессии
w_analytic = inv(
    features.T.dot(features)
).dot(
    features.T
).dot(
    y
)
print("Аналитически определённые коэффициенты \n%s" % ndprint(w_analytic))


# обучаем модель "из коробки"
reg = LinearRegression().fit(features, y)
print("Коэффициенты, вычисленные моделью sklearn \n%s" % ndprint(reg.coef_))