import numpy as np
import pandas as pd
from sklearn import cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)  # Missing data is marked with '?' which we replace by outlier value
# The 'id' column has no revelance to label, in k nearest it actually hurts classification. Comment out and see accuracy
df.drop(['id'], 1, inplace=True)    # If commenting out, add extra cokumn of data to mock_data below

X = np.array(df.drop(['class'], 1))    # Features are everything but the class column in data frame
y = np.array(df['class'])              # Labels just the class column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy of training on test data: ", accuracy)  # Comment out dropping of 'id' column to see how much it changes

# Example prediction
mock_data = [
    [4, 2, 1, 1, 1, 2, 3, 2, 1],
    [4, 2, 1, 2, 2, 2, 3, 2, 1]
    ]
example_measure = np.array(mock_data)  # Mock data without 'id' and 'class' column
# The below is to reshape the numpy array to sklearn format, reshape is a numpy function, -1 auto detects setting
example_measure = example_measure.reshape(len(example_measure), -1)
prediction = clf.predict(example_measure)
print("Prediction of example data: ", prediction)
