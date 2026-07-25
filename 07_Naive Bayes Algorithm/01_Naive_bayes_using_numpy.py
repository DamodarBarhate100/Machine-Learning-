import numpy as np

X = np.array([
    [1, 1, 0, 0], # Spam
    [1, 0, 0, 0], # Spam
    [0, 1, 1, 0], # Spam (a tricky one)
    [1, 1, 0, 0], # Spam
    [0, 0, 1, 1], # Normal
    [0, 0, 1, 0], # Normal
    [0, 1, 1, 1], # Normal (has "discount")
    [0, 0, 0, 1]  # Normal
])

y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
def fit(X,y):

    n = X.shape[1]
    m = len(y)
    phi_y = np.sum(y)/m

    phi_y_0 = np.mean(X[y==0],axis=0) + 1e-6 
    phi_y_1 = np.mean(X[y==1],axis=0) + 1e-6 
    return phi_y, phi_y_1, phi_y_0
phi_y, phi_y_1, phi_y_0 = fit(X,y)
print("Probability of Spam (y=1):", phi_y)
print("Word probabilities if Spam:", phi_y_1)
print("Word probabilities if Normal:", phi_y_0)


def predict(X,phi_y_1, phi_y_0, phi_y):
    log_prior = np.log(phi_y)
    log_likely_1 = X * np.log(phi_y_1) + (1 - X) * np.log(1 - phi_y_1)
    log_likely_0 = X * np.log(phi_y_0) + (1 - X) * np.log(1 - phi_y_0)
    score_0 = log_prior + np.sum(log_likely_0, axis=1)
    score_1= np.log(1 - phi_y) + np.sum(log_likely_1, axis=1)
    return (score_1 > score_0).astype(int)

predictions = predict(X[0:5],phi_y_1, phi_y_0, phi_y)
print(predictions)