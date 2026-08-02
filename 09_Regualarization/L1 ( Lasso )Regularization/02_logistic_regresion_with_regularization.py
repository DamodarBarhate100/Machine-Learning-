import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix

df = pd.read_csv('logistic_overfit_data.csv')

X = df.drop("target",axis=1).to_numpy()
y = df['target'].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=42,test_size=0.2)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def logistic_regression(X_train, y_train, lambda_ = 10.0):
    m = len(y_train)
    iterations = 2000
    learning_rate = 0.05
    x_with_bias = np.column_stack((X_train, np.ones(shape = m)))
    n = x_with_bias.shape[1]
    theta_j = np.zeros(shape=n)

    for i in range(iterations):
        z = x_with_bias @ theta_j
        y_prediction = sigmoid(z)
        error = y_train - y_prediction
        penalty_theta_j = np.copy(theta_j)
        penalty_theta_j[-1] = 0
        penalty_theta_j[:-1] = (lambda_/m) * np.sign(theta_j[:-1])
        gradient = (1/m) * error@x_with_bias - penalty_theta_j
        theta_j = theta_j + learning_rate * gradient

    return theta_j

theta_j = logistic_regression(X_train, Y_train, 1.0)
m = theta_j[:-1]
b = theta_j[-1]
print("\n Weights:", m)
print("Intercept:", b)

def decision_boundary(y_sigmoid):
    return (y_sigmoid >= 0.5).astype(int)

def predict(x_train, y_train, x_test, y_test, theta_j):
    x_with_bias =  np.column_stack((x_train, np.ones(shape = x_train.shape[0])))
    z = x_with_bias@theta_j
    y_sigmoid = sigmoid(z)
    y = decision_boundary(y_sigmoid)
    print("\nEvaluation Metrics On X_train  --- seen data:")
    print("Accuracy:", accuracy_score(y_train, y))
    print("Precision:", precision_score(y_train, y))
    print("Recall:", recall_score(y_train, y))
    print("Confusion Matrix:\n", confusion_matrix(y_train, y))

    x_test_with_bias =  np.column_stack((x_test, np.ones(shape = x_test.shape[0])))
    z_ = x_test_with_bias@theta_j
    y_test_sig = sigmoid(z_)
    y_test_pred = decision_boundary(y_test_sig)
    print("\nEvaluation Metrics On X_test  --- unseen data  data:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred))
    print("Recall:", recall_score(y_test, y_test_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

predict(X_train,Y_train, X_test, Y_test, theta_j)

    