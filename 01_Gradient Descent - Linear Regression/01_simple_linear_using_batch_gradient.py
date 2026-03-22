import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


X = np.array([1,2,3,4,5,6,7,8])
y = np.array([3.2,4.9,7.1,9.2,10.8,13.4,15.3,16.9])


def batch_gradient_decent(x,y):
    n = 1
    theta_j = 0.0
    theta_0 = 0.0
    iterations = 600
    learning_rate = 0.05
    m = len(y)
    for i in range(iterations):
        y_predicted = (theta_j * x) + theta_0
        errrors = y_predicted - y
        cost_func = (1/(2*m)) * np.sum(np.square(errrors))
        gradient_j = (1/m) * np.sum(errrors*x)
        gradient_0 = (1/m) * np.sum(errrors)
        theta_j = theta_j - learning_rate * gradient_j
        theta_0 = theta_0 - learning_rate * gradient_0
        if i % 50 == 0:
            print(f"Iter {i}: m={theta_j}, b={theta_0}, cost={cost_func}")    
    return theta_j,theta_0


m_actual,b_actual = batch_gradient_decent(X,y)
print("Value of slope (m) and intercept (b):")
print("slope(m):",m_actual)
print("intercept(b):",b_actual)

line = (m_actual * X) + b_actual

# Visualization
plt.scatter(X,y,color='lightgreen')
plt.plot(X,line,label='Regression Line',color='salmon')
plt.title("Simple Linear Regression using Batch Gradient Descent",fontweight='bold')
plt.xlabel("Input X",fontweight='bold')
plt.ylabel("Ouput Y",fontweight='bold')
plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()