import numpy as np
import pandas as pd

df = pd.read_csv("Dataset/gda_dataset.csv")

X = df.drop(labels='y', axis=1)
y = df['y']

def fit(X, y):
    X = np.array(X)
    y = np.array(y).flatten()
    m = len(y)
    phi = (np.sum(y))/m
    y_0 = len(y[y==0])
    y_1 = len(y[y==1])
    mu_0 = (np.mean(X[y==0],axis=0))
    mu_1 = (np.mean(X[y==1], axis=0))

    X_center = np.copy(X)
    X_center[y==0] -= mu_0
    X_center[y==1] -= mu_1
    Sigma = (X_center.T @ X_center)/m
    return phi, mu_0, mu_1, Sigma

phi, mu_0, mu_1, sigma = fit(X, y)

def predict(X, phi, mu_0, mu_1, sigma):
    x = np.array(X)
    d = x.shape[1] 
    
    sigma_det = np.linalg.det(sigma)
    constant = 1 / ( ((2 * np.pi) ** (d / 2)) * np.sqrt(sigma_det) )
    
    sigma_inv = np.linalg.inv(sigma)
    diff_0 = x - mu_0
    diff_1 = x - mu_1
    exponent_term = -0.5 * np.sum((diff_0 @ sigma_inv) *  diff_0, axis=1)
    exponent_term_1 = -0.5 * np.sum((diff_1 @ sigma_inv) * diff_1, axis=1)
    pdf_y_0 = constant * np.exp(exponent_term)
    pdf_y_1 = constant * np.exp(exponent_term_1)
    
    p_y_0 = pdf_y_0 * (1 - phi)
    p_y_1 = pdf_y_1 * phi
    return (p_y_1 > p_y_0).astype(int)

X_new = np.array([
    [1.5, 1.5],  # Point 1
    [7.0, 6.0],  # Point 2
    [4.0, 3.5],  # Point 3 
    [-2.0, 9.0]  # Point 4
])
predictions  = predict(X_new,phi, mu_0, mu_1, sigma=sigma)
print("--- Predictions on New Data ---")
for i, point in enumerate(X_new):
    print(f"Data Point {point} --> Predicted Class: {predictions[i]}")