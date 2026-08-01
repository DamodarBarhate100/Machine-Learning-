import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('L1 ( Lasso )Regularization/overfit_data.csv')
df.drop("split", axis=1, inplace=True)

X = df.drop('target', axis=1).to_numpy()
y = df['target'].to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



def lin_reg_without_regularization(x_train, y_train):
    m = len(x_train)
    n = x_train.shape[1]
    learning_rate = 0.01
    theta_j = np.zeros(shape=n)
    theta_0 = 0
    iterations = 1000
    
    for i in range(iterations):
        y_predicted = x_train @ theta_j + theta_0
        error = y_predicted - y_train

        gradient_j = (1/m) * error @ x_train
        theta_j = theta_j - learning_rate * gradient_j
        theta_0 = theta_0 - learning_rate * (1/m) * np.sum(error)
        
    return theta_j, theta_0



m_unreg, b_unreg = lin_reg_without_regularization(X_train_scaled, Y_train)

def predict(m, b, X_train, Y_train, X_test, Y_test):
    y_train_pred = X_train @ m + b
    train_mse = (1/len(Y_train)) * np.sum((y_train_pred - Y_train)**2)

    y_test_pred = X_test @ m + b
    test_mse = (1/len(Y_test)) * np.sum((y_test_pred - Y_test)**2)

    print("--- Model Performance ---")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE:     {test_mse:.2f}")

print("\nWithout Regularization:")
predict(m_unreg, b_unreg, X_train_scaled, Y_train, X_test_scaled, Y_test)
