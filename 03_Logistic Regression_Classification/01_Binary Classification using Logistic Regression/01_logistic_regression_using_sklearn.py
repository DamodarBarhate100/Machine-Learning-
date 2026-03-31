import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset from scikit-learn
data = load_breast_cancer()

# 2. Create a DataFrame for the features (X)
df = pd.DataFrame(data.data, columns=data.feature_names)

# 3. Add the target column (y) to the DataFrame
df['target'] = data.target

# 4. View the first 5 rows
print(f"Dataset Shape: {df.shape}")


log_reg = LogisticRegression(max_iter=3000)

X = df.drop('target', axis=1)  
y = df['target']

x_train, x_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
log_reg.fit(x_train,y_train)
print("theta:",log_reg.coef_)
print("intercept:",log_reg.intercept_)

y_predicted = log_reg.predict(x_test)
print("\n Predicted values")
print(y_predicted[:10])
print("\n Actual Values:")
print(y_test[:10])


# 1. Calculate Accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"\nAccuracy: {accuracy:.2f}")  

# 2. Confusion Matrix (True Positives, False Negatives, etc.)
conf_matrix = confusion_matrix(y_test, y_predicted)
print("\nConfusion Matrix:")
print(conf_matrix)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predicted))