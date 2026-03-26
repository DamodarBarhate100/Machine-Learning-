import numpy as np

# Input features (4 samples, 2 features each)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target labels (OR logic)
y = np.array([0, 1, 1, 1])

def perceptron(x,y):
    m = len(y)
    learning_rate  = 0.5
    iterations = 500
    x_with_bias = np.column_stack((np.ones(shape=m),x))
    n = x_with_bias.shape[1]
    theta_j = np.zeros(shape=n)
    for i in range(iterations):
        z = theta_j@x_with_bias.T
        y_predicted = np.where(z>0,1,0)
        # its means that if z>0 then give 1 else give 0
        errors = y - y_predicted
        #  actual - predicted
        gradient = errors @ x_with_bias
        theta_j = theta_j + learning_rate*gradient
    
    return theta_j

theta = perceptron(X,y)
print("Intercept:",theta[0])
print("Weights:",theta[1:])

print("If we predict values based on the seen data x:")
x_with_bias = np.column_stack((np.ones(shape=len(y)),X))
z = theta@x_with_bias.T
y_predict = np.where(z>0,1,0)
print(y_predict)