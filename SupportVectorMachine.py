import numpy as np
import pandas as pd
from sklearn import cross_validation, svm

'''
The same code as KNN using scikit-learn, only thing changed is using Support Vector Machine class as classifier
'''

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)    # Same problem as with KNN, adding this column significantly decreases accuracy!!

X = np.array(df.drop(['class'], 1))    # Features
y = np.array(df['class'])              # Labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy of training on test data: ", accuracy)

# Example prediction
mock_data = [
    [4, 2, 1, 1, 1, 2, 3, 2, 1],
    [4, 2, 1, 2, 2, 2, 3, 2, 1]
    ]
example_measure = np.array(mock_data)  # Mock data without 'id' and 'class' column
example_measure = example_measure.reshape(len(example_measure), -1)
prediction = clf.predict(example_measure)
print("Prediction of example data: ", prediction)
