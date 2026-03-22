import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5,6,7,8])
y = np.array([3.2,4.9,7.1,9.2,10.8,13.4,15.3,16.9])

def stochastic_gradient_decent_corrected(x, y):
    theta_j = 0.0  # Slope (m)
    theta_0 = 0.0  # Intercept (b)
    epochs = 600  # Now represents 600 full passes over the data
    learning_rate = 0.005 # Reduced for better stability
    m = len(y)
    
    # Combine X and y for easy simultaneous shuffling
    data = np.c_[x, y]
    
    for i in range(epochs):
        # Shuffle the data at the start of each epoch for true SGD
        np.random.shuffle(data)
        x_shuffled = data[:, 0]
        y_shuffled = data[:, 1]
        
        # Iterate over all samples (one update per sample)
        for k in range(m):
            # Select the single sample
            x_i = x_shuffled[k]
            y_i = y_shuffled[k]
            
            # Prediction and Error
            y_predicted_i = (theta_j * x_i) + theta_0
            error_i = y_predicted_i - y_i
            
            # SGD Gradient Update (based on ONE sample)
            gradient_j = error_i * x_i
            gradient_0 = error_i
            
            # Parameter Update
            theta_j = theta_j - learning_rate * gradient_j
            theta_0 = theta_0 - learning_rate * gradient_0

        # Calculate cost after the full epoch (using the current parameters and ALL data)
        y_predicted_total = (theta_j * x) + theta_0
        errors = y_predicted_total - y
        cost_func = (1/(2*m)) * np.sum(np.square(errors))
        
        if i % 50 == 0:
            print(f"Epoch:{i}: m={theta_j:.4f}, b={theta_0:.4f}, Cost={cost_func:.4f}")

    # Return the scalar values from the final one-element arrays
    return theta_j, theta_0


m_actual, b_actual = stochastic_gradient_decent_corrected(X, y)
print("\n--- Final Results ---")
print("Value of slope (m) and intercept (b):")
print(f"slope(m): {m_actual:.4f}")
print(f"intercept(b): {b_actual:.4f}")

line = (m_actual * X) + b_actual

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='lightgreen', label='Data Points', edgecolors='black')
plt.plot(X, line, label='Regression Line (SGD Corrected)', color='salmon', linewidth=2)
plt.title("Simple Linear Regression using Stochastic Gradient Descent", fontweight='bold')
plt.xlabel("Input X", fontweight='bold')
plt.ylabel("Ouput Y", fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()