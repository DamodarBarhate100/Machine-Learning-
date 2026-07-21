import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('terrain_sensor.csv')
X = df[df.columns[:-1]]
y = df["Target"]

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
scaler = StandardScaler()
x_scaled= scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

softmax_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax_reg.fit(x_scaled, y_train)

print("\nCoefficient:", softmax_reg.coef_)
print("Intercept:", softmax_reg.intercept_)

print("\n Predictions:")
y_pred = softmax_reg.predict(x_test_scaled)
print(y_pred)

print(f"\nAccuracy: {accuracy_score(y_true=y_test, y_pred=y_pred):.4f}")
print(f"Precision: {precision_score(y_true=y_test, y_pred=y_pred, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_true=y_test, y_pred=y_pred, average='weighted'):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_true=y_test, y_pred=y_pred))