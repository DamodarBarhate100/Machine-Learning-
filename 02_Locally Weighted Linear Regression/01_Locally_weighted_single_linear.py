import numpy as np

np.random.seed(42)

m = 100  # Number of examples

# Generate 100 points between -3 and 3
X = np.linspace(-3, 3, 100)

# Generate Y values using a non-linear function (Sine wave) + random noise
# Shape: (100,)
Y = np.sin(X) + np.random.normal(0, 0.1, 100)

print("\nX values:")
print(X[:10])
print("\nY values:")
print(Y[:10])


#  at what value you want to predict the output
x_i = np.array([2.5])
tau = 0.5
X_with_bias = np.column_stack((np.ones(shape=m),X))

expression = (np.square(X - x_i)) / (2 * (tau**2))
w_i = np.exp(-expression)
W = np.diag(w_i)
print(X_with_bias.shape)
print(W.shape)
theta_x_i = np.linalg.inv(X_with_bias.T @ W @ X_with_bias) @ (X_with_bias.T @ W @ Y)

print(theta_x_i)

print("\n Predicted Value for the x_i is:")
x_i_with_bias = np.column_stack((np.ones(shape=x_i.shape),x_i))
y_predicted = x_i_with_bias @ theta_x_i
print(y_predicted)