import numpy as np

# Set seed for reproducibility
np.random.seed(23)
m = 80

X_hp = np.random.uniform(70, 200, m) 
X_wt = np.random.uniform(2000, 4500, m)

X_train = np.column_stack((X_hp, X_wt))

Y_target = (40 + np.random.normal(0, 3, m) 
            - 0.08 * X_hp 
            - 0.003 * X_wt)

Y_target = np.clip(Y_target, 10, 45)


print(f"X_train shape: {X_train.shape}")
print(f"Y_target shape:{Y_target.shape}")
print(f"X_train Values: {X_train[:10]}")
print(f"Y_target Values: {Y_target[:10]}")

x_i = np.array([193, 2960])
tau = 500
expression = np.sum(np.square(X_train - x_i),axis=1)/ (2* (tau**2))
w_i = np.exp(-expression)
W = np.diag(w_i)
X_train_with_bias = np.column_stack((np.ones(m),X_train))
print(W.shape)
print(X_train.shape)
term_1 = np.linalg.inv(X_train_with_bias.T @ W @ X_train_with_bias)
term_2 = (X_train_with_bias.T @ W @ Y_target)
theta_x_i = term_1 @ term_2
print(theta_x_i)

x_i_with_bias = np.insert(x_i,0,1)
y_predicted = x_i_with_bias @ theta_x_i
print("\n Predicted Values:")
print(y_predicted)