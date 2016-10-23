import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')

''' First we show Mean Shift works on small scale data '''
centres = [[2, 2, 2],
           [4, 5, 6],
           [13, 11, 11]]   # Centre of clusters
X, _ = make_blobs(n_samples=100, centers=centres, cluster_std=1.5)  # Generate random data points to cluster

clf = MeanShift()
clf.fit(X)
labels = clf.labels_    # Same as KMean labels, [ 0 0 1 0 2 ...] each data point in X labelled with cluster number
cluster_centers = clf.cluster_centers_
cluster_count = len(np.unique(labels))
# print("Cluster centres: ", cluster_centers)
# print("Number of clusters: ", cluster_count)

colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the sample data points
for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

# Add the cluster points
ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], marker="x", color='k', s=30,
           linewidths=5, zorder=10)
# plt.show()

''' We now use Mean Shift on the titanic dataset.

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
original_df = pandas.DataFrame.copy(df)     # Make a deep copy
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

clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_
cluster_count = len(np.unique(labels))
original_df['cluster_group'] = np.nan

for i in range(len(X)):
    # We attach the clustered labels to rows in dataframe. iloc gives index on dataframe row
    original_df['cluster_group'].iloc[i] = labels[i]

'''
With KMeans we passed no of clusters as 2 and expected the algo to classify into survivors and not.
Here we let the algo tell us how many groups it feels the data has. Once it has claffified it, we can then look deeper
into how the data was classified and see if we can spot pattern
'''
survival_rates = {}     # cluster group number: survival rate
for i in range(cluster_count):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]   # New df only containing data from ith cluster
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate
print(survival_rates)

# Re-running the algorithm can produce different results, maybe even more/less clusters next time. So remember this
# when coding any hard-coded values for cluster count on analysis

# Analysis of data
for i in survival_rates:
    print("CLuster group: ", i)
    # print(original_df[(original_df['cluster_group'] == i)])  # To see the df associated with each cluster group
    print(original_df[(original_df['cluster_group'] == 2)].describe())  # To get Stats on column values of cluster
