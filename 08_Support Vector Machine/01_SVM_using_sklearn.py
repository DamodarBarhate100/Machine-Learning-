import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Dataset/robot_quality_dataset.csv")
X = df.drop(labels="quality", axis=1)
y =df["quality"]
print("\nExploring the dataset:")
print(df.head())
print(df.info())
print(df.describe())

print("\nChecking the balance of the class:")
print(len(df[df["quality"] == 0])/len(df) * 100)

# splitting the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=42, test_size=0.2)

scaler = StandardScaler()
svm_linear = SVC(kernel='linear', C=1, gamma='scale')
svm_poly = SVC(kernel='poly', C=1,gamma='scale')
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')

x_scaled= scaler.fit_transform(X=X_train)
X_test_scaled = scaler.transform(X_test)

svm_linear.fit(x_scaled, Y_train)
svm_poly.fit(x_scaled, Y_train)
svm_rbf.fit(x_scaled, Y_train)

predictions_rbf  = svm_rbf.predict(X_test_scaled)
predictions_poly  = svm_poly.predict(X_test_scaled)
predictions_liner  = svm_linear.predict(X_test_scaled)

print("\n Comparing different types of support vector machines:")

print("\n Radial Basis Function")
print("Accuracy :", accuracy_score(Y_test, predictions_rbf))
print("Precision:", precision_score(Y_test, predictions_rbf))
print("Recall   :", recall_score(Y_test, predictions_rbf))
print(confusion_matrix(Y_test, predictions_rbf))
print("\n Classification Report",classification_report(Y_test, predictions_rbf))

print("\n Polynomial Function")
print("Accuracy :", accuracy_score(Y_test, predictions_poly))
print("Precision:", precision_score(Y_test, predictions_poly))
print("Recall   :", recall_score(Y_test, predictions_poly))
print(confusion_matrix(Y_test, predictions_poly))
print("\n Classification Report",classification_report(Y_test, predictions_poly))


print("\n Linear Function")
print("Accuracy :", accuracy_score(Y_test, predictions_liner))
print("Precision:", precision_score(Y_test, predictions_liner))
print("Recall   :", recall_score(Y_test, predictions_liner))
print(confusion_matrix(Y_test, predictions_liner))
print("\n Classification Report",classification_report(Y_test, predictions_liner))

# The RBF kernel implicitly maps the data into a very high-dimensional (theoretically infinite-dimensional) 
# feature space, allowing the SVM to learn nonlinear decision boundaries without explicitly computing those new features