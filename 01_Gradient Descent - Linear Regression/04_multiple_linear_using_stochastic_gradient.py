import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

y = np.array([65, 90, 50, 110, 70, 85, 62, 95, 88, 58])   # in lakhs

#  Feature scaling ---  Normalization
x_mean = X.mean(axis=0)
x_std = X.std(axis=0)
x_scaled = (X-x_mean)/x_std

y_mean = y.mean()
y_std = y.std()
y_scaled = (y-y_mean)/y_std

def stochastic_gradient_decent(x,y):
    iterations = 500
    learning_rate = 0.01
    m = len(y)
    x_with_bias = np.column_stack((np.ones(shape=m),x))
    n = x_with_bias.shape[1]
    theta_j = np.zeros(shape=n)
    data = np.c_[x_with_bias,y]
    for i in range(iterations):
        np.random.shuffle(data)
        x_shuffled = data[:,0:-1]
        y_shuffled = data[:,-1]
        for j in range(m):
            x_i = x_shuffled[j]
            y_i = y_shuffled[j]
            y_predicted_i = theta_j@x_i.T
            errors_i = y_predicted_i - y_i
            gradient = errors_i * x_i
            theta_j = theta_j - learning_rate * gradient
        y_predicted = theta_j @ x_with_bias.T
        errors = y_predicted - y
        cost = (1/(2*m)) * np.sum(np.square(errors))
        if(i%20 == 0):
            print("theta_j:{} cost:{} i:{}".format(theta_j,cost,i))

    return theta_j[1:] ,theta_j[0]
m_scaled,b_scaled = stochastic_gradient_decent(x_scaled,y_scaled)


# # Unscaling the slopes and intercept
m_actual = m_scaled * (y_std/x_std)
b_actual = (b_scaled * y_std) + y_mean - np.dot(m_actual, x_mean)
print("\nValues of slopes (m) and intercept (b):")
print("slope(m):",m_actual)
print("intercept(b):",b_actual)

# predicting new values on seen data:
X_new = np.array([
    [1400, 3, 5.0],
    [2000, 4, 2.0],
    [1100, 2, 10.0]
])
print("\n Predicted Values:")
y_predicted = b_actual + (X_new @ m_actual)
print(y_predicted)