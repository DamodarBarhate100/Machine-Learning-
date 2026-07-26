from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('terrain_sensor.csv')
X = df.drop(labels='Target', axis=1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)
print("\n Class prior:", naive_bayes.class_prior_)
print("\n Mean :", naive_bayes.theta_)
print("\n Variance :", naive_bayes.var_)

predictions = naive_bayes.predict(x_test)
print("\n Predictions:", predictions)

print("\n Metrics:")
print("Accuracy Score:",accuracy_score(y_test, predictions))
print("\nPrecision Score:", precision_score(y_test, predictions, average='weighted'))
print("Recall Score:", recall_score(y_test, predictions, average='weighted'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

