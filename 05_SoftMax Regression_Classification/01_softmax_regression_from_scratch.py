import numpy as np
import pandas as pd

df = pd.read_csv("terrain_sensor.csv")
print("Terrain Sensor Dataset:\n", df)

X = df[df.columns[:-1]]
y = df["Target"]


def softmax(z):
    temp = np.exp(z - np.max(z,axis=1,keepdims=True))
    return temp/np.sum(temp, keepdims=True, axis=1)

def batch_gradient_descent(x,y):
    iterations = 300
    m = len(y)
    x_with_bias = np.column_stack((np.ones(shape=m),x))
    n = x_with_bias.shape[1]
    theta_j = np.zeros(shape=(n,3))
    learning_rate = 0.05
    y_encoded = pd.get_dummies(y,prefix='en', dtype=int).to_numpy()
    for i in range(iterations):
        logits = (x_with_bias @ theta_j)
        y_predicted = softmax(logits)
        loss = -np.mean(np.sum(y_encoded * np.log(y_predicted + 1e-8), axis=1))
        errors = y_predicted - y_encoded

        # gradient 
        gradient = (1/m) * x_with_bias.T @ errors
        theta_j = theta_j - learning_rate * gradient

        if i%30 == 0:
            print("loss convergence:",loss)

    return theta_j


theta_j = batch_gradient_descent(X,y)
print("\nWeights:",theta_j[1:])
print("Bias:",theta_j[0])


print("\n Predicting on the new unseen data:")
test_data = pd.read_csv("test_terrain.csv")
x_test = test_data[test_data.columns[:-1]]
X_test_with_bias = np.column_stack((np.ones(shape=x_test.shape[0]),x_test))

logits = (X_test_with_bias @ theta_j)
probs = softmax(logits)
prediction = np.argmax(probs,axis=1) 

print(f"Predicted Class: {prediction}")

y_test_actual = test_data["Target"].to_numpy()

accuracy = np.mean(prediction == y_test_actual) * 100
print(f"Model Accuracy: {accuracy:.2f}%")