import numpy as np

np.random.seed(42)
m = 100 
X = np.random.uniform(-3, 3, (m, 2))
Y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, m)

# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------
# Add bias to training set ONCE
X_train_bias = np.column_stack((np.ones(shape=m), X))

# The points we want to predict for (First 5 points of X)
query_points = X[0:5, :] 
tau = 0.2

# ---------------------------------------------------------
# FUNCTION: Predict for a SINGLE point
# ---------------------------------------------------------
def predict_lwlr(query_point, X_train, Y_train, X_train_bias, tau):
    # 1. Calculate Weights
    # Query point is (2,) vs X_train (100, 2) -> Broadcasing works here
    dist_squared = np.sum(np.square(X_train - query_point), axis=1)
    w_i = np.exp(-dist_squared / (2 * (tau**2)))
    W = np.diag(w_i)
    
    # 2. Calculate Theta (Local to this specific query_point)
    theta = np.linalg.inv(X_train_bias.T @ W @ X_train_bias) @ (X_train_bias.T @ W @ Y_train)
    
    # 3. Predict
    query_point_bias = np.insert(query_point, 0, 1)
    prediction = query_point_bias @ theta
    return prediction

# ---------------------------------------------------------
# LOOP THROUGH EACH QUERY POINT
# ---------------------------------------------------------
print("Predictions for the first 5 points:")

for i in range(len(query_points)):
    # Extract one point (shape: (2,))
    single_point = query_points[i]
    
    # Get prediction
    pred = predict_lwlr(single_point, X, Y, X_train_bias, tau)
    
    # Compare with actual Y (just to see accuracy)
    actual = Y[i]
    
    print(f"Point {i}: Predicted={pred:.4f}, Actual={actual:.4f}")