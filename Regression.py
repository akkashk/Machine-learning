"""
This file contains code in using scikit-learn along with other modules to do linear regression on data.
"""

import datetime
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import quandl
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

style.use('ggplot')

df = quandl.get("WIKI/GOOG")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]  # Adjust the columns
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100  # Add new column
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # Adjust columns to display again

forecast_col = 'Adj. Close'      # The column to predict
df.fillna(-99999, inplace=True)  # To replace NaN with outlier value

forecast_out = int(math.ceil(0.01 * len(df)))         # No of days into future to predict, here we do 1% (0.01%)
print("No of days into future predicting: ", forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)  # The shift() moves the table by forecast amount (i.e. predicting)

X = np.array(df.drop(['label'], 1))  # X = features, will be everything but the label column
X = preprocessing.scale(X)           # Normalising the input data
X_lately = X[-forecast_out:]         # This the the amount we're predicting, we don't have y value for these
X = X[:-forecast_out]                # Reduce feature set, removing data we don't have y value for

df.dropna(inplace=True)     # We drop rows without attached labels, in data frame it will be Na
y = np.array(df['label'])   # y = Label, the label column

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)  # Use 20% of data as test

clf = LinearRegression()                # Classifier
clf.fit(X_train, y_train)               # Train classifier
accuracy = clf.score(X_test, y_test)    # Test classifier

# print("Accuracy of prediction on test data: ", accuracy)
forecast_set = clf.predict(X_lately)    # We use predict to get y value(s), it takes a single/array of values to predict
print("Predicted y values: ", forecast_set)
print("Accuracy of classifier: ", accuracy)
print("Number of days ahead of time predicting: ", forecast_out)

# Below is for plotting the data on a graph
df['Forecast'] = np.nan         # Create a Forecast column to fill data
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:          # Loop is 'hacking' a way to get Date onto x-axis
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # Below line sets NaN for all columns except the Forecast column which is set to predicted value
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# print(df.tail())      # Uncomment this line to see what last loop line does
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()              # The actual values and forecast value are shown in graph with different legend

'''
Pickling: Every time we run this, we are training a classifier and then predicting. This is fine for small data sets
such as this but if we have gigabytes of data then training classifier every time would be inefficient. So we can
pickle the trained classifier object (serialise it) and then read back the object from disk and use this whenever.
NOTE: If we are doing so then it would be good to retrain the classifier every now and then to keep it accurate.
'''
