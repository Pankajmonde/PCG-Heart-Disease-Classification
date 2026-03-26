import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris

# Load dataset (for demonstration, using iris dataset)
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual SVM models
svm1 = svm.SVC(kernel='linear', probability=True)
svm2 = svm.SVC(kernel='rbf', probability=True)
svm3 = svm.SVC(kernel='poly', degree=3, probability=True)

# Create an ensemble of SVMs using voting classifier
voting_clf = VotingClassifier(estimators=[('linear', svm1), ('rbf', svm2), ('poly', svm3)], voting='soft')

# Train the ensemble model
voting_clf.fit(X_train, y_train)

# Evaluate the model
accuracy = voting_clf.score(X_test, y_test)
print(f'Ensemble SVM model accuracy: {accuracy:.2f}')