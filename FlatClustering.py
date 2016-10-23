import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas

style.use('ggplot')

''' We use clustering on a small input to see how it works '''
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()

# clf = KMeans(n_clusters=2)
# '''
# If the number of centroids we want == number of sample point, then each feature set will be a centroid
# If the number if centroids > number of sample points, we get an error message (most logical thing to do)
# '''
# clf.fit(X)
# centroids = clf.cluster_centers_  # Array containing coordinates of centroids, [[ 1.16666667, 1.46666667] [ 8.5, 9.5]]
# print(centroids)
# labels = clf.labels_    # Labels will be an array of labels that got assigned to our input feature sets, [0 0 1 0 1]
# print(labels)
#
# colours = 2 * ['g.', 'r.', 'c.', 'b.', 'k.']
# for i in range(len(X)):
#     # The data, since labela will be 0/1 here, we index into colours arr
#     plt.plot(X[i][0], X[i][1], colours[labels[i]], markersize=20)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
# plt.show()

''' Next we use the titanic dataset to try and find any clustering on feature sets (people information) and cluster
them. We will pass a parameter to cluster them into two groups and we will see whether it clusters them by whether
people survived or not. We do not know if this is how it will cluster! It contains the following data:

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pandas.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)  # Srop columns that don't add value
df.fillna(0, inplace=True)  # Here NaN only occur on columns where we count such as no. of lifeboats, so we replace by 0


# Some of the important columns (such as Cabin/Sex) have non-numerical data. ML needs numerical data
def handle_non_numerical_data(df):
    columns = df.columns.values
    for col in columns:         # Go through each column
        text_digit_vals = {}    # Have a mapping dictionary ready for that column

        def convert_to_int(val):    # Returns the numerical mapping of text value from mapping dictionary
            return text_digit_vals[val]

        if df[col].dtype != np.int64 and df[col].dtype != np.float64:   # If column is not numerical
            col_contents = df[col].values.tolist()                      # Convert to a list
            unique_elements = set(col_contents)                         # Get set of unique elements
            x = 0                                                       # Mapping value counter
            for unique in unique_elements:
                if unique not in text_digit_vals:                       # If value not in mapping dictionary
                    text_digit_vals[unique] = x                         # Assign numerical mapping value
                    x += 1                                              # Increment mapping counter

            df[col] = list(map(convert_to_int, df[col]))                # Set new column values
    return df

df = handle_non_numerical_data(df)

df.drop(['pclass'], 1, inplace=True)    # Improves prediction, found by trial and error

# Before we use algo, drop survived column as we want this to be label, convert data to float
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)
# In sci-kit learn, the first centroid it finds will be given label 0 and the next 1 and so on...
# So if we are to compare known labels to values, we do not know which centroid got which label, here survived may be 0
# and not be 1, but we may have it other way around when comapring. So accuracy of 20% means actually 80% but we just
# have the labels wrong way around. Again, KMeans doesn't know what to classify by, it just finds clusters which may/not
# correspond to the labels we were hoping to find

correct = 0
for i in range(len(X)):
    predict = np.array(X[i].astype(float))
    predict = predict.reshape(-1, len(predict))
    prediction = clf.predict(predict)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(X))  # 49%, so far it hasn't classified by survivors, without preprocessing. With prepro, its ~70%
# We can drop or not drop columns and see how it changes accurary of predicion
