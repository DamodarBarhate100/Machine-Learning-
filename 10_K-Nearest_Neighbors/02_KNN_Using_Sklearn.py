import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors=5)
scaler_ = StandardScaler()
df = pd.read_csv('multi_terrain_data.csv')

X = df.drop('terrain_class', axis=1)
Y = df['terrain_class']

X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, random_state=42, test_size=0.2)

x_train_scaled = scaler_.fit_transform(X_train)
x_test_scaled = scaler_.transform(X_test)

Knn.fit(x_train_scaled, Y_train)

print("\n Predictions on test data set:")
y_predictions = Knn.predict(x_test_scaled)
print(y_predictions)

print("\n ----------------------- Evaluation Metrics -----------------------------")
print("Accuracy:", accuracy_score(y_pred=y_predictions, y_true=Y_test))
print("Precision Score:", precision_score(y_pred=y_predictions, y_true=Y_test, average='weighted'))
print("Recall:", recall_score(y_pred=y_predictions, y_true=Y_test, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_pred=y_predictions, y_true=Y_test))
