# Linear Regression from Scratch: The Power of L1 Regularization (Lasso)

## Project Overview
When a dataset contains many irrelevant features (noise) and very few informative features (signal), standard linear regression will **overfit** by memorizing the noise. 

This project explores three states of a machine learning model:
1. **Unregularized Model:** Memorizes noise, resulting in high Test Error (Overfitting).
2. **L1 Regularization (Lasso):** Shrinks weights using absolute values, driving the 17 useless noise features precisely to `0.0` while preserving the 3 true signals.

## The Dataset
The synthetic dataset (`overfit_data.csv`) is explicitly engineered to cause overfitting:
* **Total Features ($n$):** 20
* **True Signals:** 3 features have actual mathematical weight.
* **Pure Noise:** 17 features are completely random noise.
* **Dataset Size ($m$):** 50 total rows (30 Train, 20 Test). 

Because the number of features (20) is dangerously close to the number of training rows (30), the unregularized model perfectly memorizes the training data.

## Mathematical Foundations (Implemented in NumPy)

### 1. Without Regularization (Standard Gradient Descent)
The standard gradient update calculates the error and adjusts the weights ($\theta$) to minimize the Mean Squared Error (MSE).
$$\text{gradient}_j = \frac{1}{m} X^T (\hat{y} - y)$$
$$\theta_j = \theta_j - \alpha \cdot \text{gradient}_j$$

### 2. With L1 Regularization (Lasso)
To prevent overfitting, a penalty term ($\lambda$) is added to the gradient. L1 applies this penalty to the **absolute value** of the weights. The derivative of an absolute value is its sign, making the NumPy implementation highly efficient using `np.sign()`.
$$\text{gradient}_j = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} \text{sign}(\theta_j)$$
$$\theta_j = \theta_j - \alpha \cdot \text{gradient}_j$$

*(Note: The bias term $\theta_0$ is intentionally excluded from the regularization penalty, as shifting the regression line up or down does not cause overfitting).*

## Execution & Results

Prior to training, the feature matrices (`X_train` and `X_test`) are scaled using `StandardScaler` to ensure the regularization penalty ($\lambda$) treats all features fairly, regardless of their native units.

### Unregularized Model Performance
```text
Training MSE: 10.69
Test MSE:     45.61
```
### Regularized Model Performance
```text
Training MSE: 27.32
Test MSE:     30.81
```
*(Result: By intentionally increasing the Training Error (refusing to memorize the noise), the Test Error drops dramatically from 45.61 to 30.81. The model successfully generalized to unseen data. )*
