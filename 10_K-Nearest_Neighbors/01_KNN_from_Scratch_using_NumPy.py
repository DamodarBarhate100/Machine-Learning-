import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("terrain_data.csv")
print(df.head())
X = df.drop("target", axis=1).to_numpy()
Y = df['target'].to_numpy().reshape(-1,1)


X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(x_test)

def knn_predict(x_train, y_train, x_test, k=0):
    predictions = []
    for i in range(len(x_test)):
        single_test_point = x_test[i]
        distances = np.sqrt(np.sum(np.square(x_train - single_test_point), axis=1))

        distances_inx = np.argsort(distances)[:k]   
        labels = y_train[distances_inx]
        unique_labels, counts = np.unique(labels, return_counts=True)
        winning_index = np.argmax(counts)
        winning_label = unique_labels[winning_index]
        predictions.append(winning_label)
    return np.array(predictions)


for i in range(1 ,10):
    y_pred = knn_predict(x_train_scaled, y_train, x_test_scaled, i)
    print("\n ----------------------- Evaluation Metrics ---------------------------------")
    print("\n For K=:", i)
    print("\n Predictions:")
    print(y_pred)
    

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision Score:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n ----------------------- ---------------------------------")

