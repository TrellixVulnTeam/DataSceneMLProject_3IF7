# Метрики качества линейной регрессии

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Сначала загрузим данные эксперимента, датасет с ценами на дома в Бостоне
boston_dataset = load_boston()
features = boston_dataset.data
y = boston_dataset.target
reg = LinearRegression().fit(features, y)

# Теперь получим два вектора – предказанное значение  y^  и истинное значение  y :
y_pred = reg.predict(features) # предсказанное значение
y_true = y # истинное значение



print("MAE = %s" % mean_absolute_error(
    reg.predict(features), y)
)

"""
Mean Squared Error (MSE) - это базовая метрика для определения качества линейной регрессии

Для каждого предсказанного значения  y^i  мы считаем квадрат отклонения от фактического значения и считаем среднее по полученным величинам
"""



mse = mean_squared_error(y_true, y_pred)

print('MSE = %s' % mse)



print("r2_score = %s" % r2_score(y_true, y_pred))