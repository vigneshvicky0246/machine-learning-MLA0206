import numpy as np
import pandas as pd

#"Importing the dataset



# divide the dataset into concepts and targets. Store the concepts into X and targets into y.
dataset = pd.read_csv("breastcancer.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values 

#Splitting the dataset into the Training set and Test  
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 2)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#Logistic Regression (LR) classifier model

#Display the results (confusion matrix and accuracy)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy Score:confusion matrix')
accuracy_score(y_test, y_pred)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
