import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
X = np.array([
    [1400, 3, 5.0],
    [2000, 4, 2.0],
    [1100, 2, 10.0],
    [2500, 4, 3.0],
    [1600, 3, 8.0],
    [1800, 3, 6.0],
    [1550, 2, 4.0],
    [2200, 4, 7.0],
    [1700, 3, 2.5],
    [1300, 2, 9.0]
])

y = np.array([65, 90, 50, 110, 70, 85, 62, 95, 88, 58])

lin_reg.fit(X,y)
print("m:",lin_reg.coef_)
print("b:",lin_reg.intercept_)
y_predicted = lin_reg.predict(X)
print(y_predicted)


