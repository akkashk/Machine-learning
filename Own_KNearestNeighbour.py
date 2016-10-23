from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import random
import warnings

style.use('fivethirtyeight')


def k_nearest_neighbours(data, predict, k=3):
    """
    :param data: Dictionary of labels and set of features for each label
    :param predict: The feature for we need to classify
    :param k: The number of neighbours
    :return: The classified label
    """
    if len(data) >= k:  # If number of labels is >= number of neighbours to test
        warnings.warn('K is set to a value less than total voting groups!')  # We haven't tested for all possible labels
    distances = []
    for label in data:
        for featureList in data[label]:
            # The below line gives most intuitive way for what we are calculating with data and test point (predict)
            # euclidean_distance = sqrt( (feature[0]-predict[0])**2 + (feature[1]-predict[1])**2 )
            euclidean_distance = np.linalg.norm(np.array(featureList) - np.array(predict))  # Built-in fast version of above
            distances.append((euclidean_distance, label))

    # Sort all distances and choose top k elements, get those labels (tup[1])
    votes = [tup[1] for tup in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    '''
    Counter.most_common: Given a list of values, the Counter class analyses it to produce a list of tuples.
    Each tuple consists of (value, count) from the list.
    most_common() takes an argument n and returns the list of top n tuples. If n < length of analysed data, returns
    available data.
    '''
    return vote_result

'''Trial dataset to test K nearest neighbuours with custom data'''
dataset = {    # The keys represent the labels of classification and sub-list represnets list of features for that label
    'k': [[1, 2], [2, 3], [3, 1]],
    'r': [[6, 5], [7, 7], [8, 6]]
    }
# for label in dataset:   # To show dataset
#     for feature in dataset[label]:
#         plt.scatter(feature[0], feature[1], s=30, color=label)      # s argument is size of points
# plt.show()

# new_feature = [5, 7]    # The new feature we want to classify and attach a label from existing label in dataset
# result = k_nearest_neighbours(dataset, new_feature, k=3)
# print("Classification of feature: label ", result)


'''Testing our K nearest neighbours algorithm with the breast cancer data to compare with sci-kit learn'''
df = pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
# Convert all data to float, full_data is list of lists (each is a row of data from input file, labels and features)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)   # Shuffles order of lists and stores in same variable

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size * len(full_data))]    # Split the data, first 80% train
test_data = full_data[-int(test_size * len(full_data)):]     # Split the data, last 20% train

# Populate the train_set and test_set from train_data and set_data
for dataRow in train_data:
    label = dataRow[-1]  # Get label attached to each row of data and use this as key in training dictionary
    train_set[label].append(dataRow[:-1])   # Append row of data without label to training dictionary

# Do same as above for train_set
[test_set[dataRow[-1]].append(dataRow[:-1]) for dataRow in test_data]

correct = 0
total = 0
for label in test_set:
    for featureList in test_set[label]:
        # The train_set is the bunch of features with attached labels and for each time we want to classify new features
        # we simply pass this as an argument. Training is just having labelled data points to compare test point to!
        vote = k_nearest_neighbours(train_set, featureList, k=5)
        if label == vote:
            correct += 1
        total += 1
print("Accuracy: ", correct / total)


