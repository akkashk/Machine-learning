import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import random


style.use('ggplot')

centers = random.randrange(2, 8)
print(centers)
X, y = make_blobs(n_samples=50, centers=3, n_features=2)
# X, y = make_blobs(n_samples=15, centers=centers, n_features=2, cluster_std=0.5)

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11],
#               [8, 2],
#               [10, 2],
#               [9, 3], ])

# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()
colours = 10 * ['g', 'r', 'c', 'b', 'k']


class MeanShift:
    def __init__(self, radius=4):
        self.radius = radius
        self.centroids = {}

    def fit(self, data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]  # Each data point becomes a centroid

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))  # Converted to tuple for future

            # The centroid list is not in any order, sorting it gives order so we can directly compare to original easy
            uniques = sorted(list(set(new_centroids)))     # To create set, we can't use numpy array, hence why tuples
            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])        # Repopulate centroids with newer version

            optimised = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]): # Earlier we sorted list to get ordering back
                    optimised = False
                if not optimised:   # Even if one centroid moves, need to recalculate all
                    break

            if optimised:       # If no centroid moves, break while loop
                break

        self.centroids = centroids

    def predict(self, data):
        pass


class DynamicMeanShift:
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step  # Number of radius steps for bandwidths, smaller value = larger steps
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        if self.radius is None:
            all_data_centroid = np.average(data, axis=0)           # Find average of all data
            all_data_norm = np.linalg.norm(all_data_centroid)      # Find magnitude of centre point
            self.radius = all_data_norm / self.radius_norm_step    # We work out radius of each step of bandwidth
        weights = [i for i in range(self.radius_norm_step)][::-1]  # Weights to apply in desc. order for each bandwidth

        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]  # Each data point becomes a centroid

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:   # When feature set is comparing centroid to itself
                        distance = 0.00000001
                    # We see how many steps we took to get to distance from centroid, if we took small steps, we would
                    # have low index in weights list and so higher weight would be given to this featureset
                    weight_index = int(distance / self.radius)
                    # If distance > maximum radius allowed, then weight index is maximum element in weights list
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    # Change the ** x, x value for performance!!!
                    to_add = (weights[weight_index] * 2) * [featureset]  # We're using squared weights to weigh feature
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))  # Converted to tuple for future

            # The centroid list is not in any order, sorting it gives order so we can directly compare to original easy
            uniques = sorted(list(set(new_centroids)))     # To create set, we can't use numpy array, hence why tuples

            # We're doing the below step since we greatly increased the feature sets included in bandwidth and so we
            # may have two centroids very close to each other
            to_pop = []
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass    # To skip condition where we're testing same centroid
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:     # Within one bandwidth step
                        # Converge onto one centroid
                        to_pop.append(ii)
                        break
            # Since we can't modify a list whilst itearting through it, need separate loop to modify list
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)
            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])        # Repopulate centroids with newer version

            optimised = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):  # Earlier we sorted list to get ordering back
                    optimised = False
                if not optimised:   # Even if one centroid moves, need to recalculate all
                    break

            if optimised:       # If no centroid moves, break while loop
                break

        self.centroids = centroids

        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        for featureset in data:
            distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in centroids]
            classification = distances.index(min(distances))
            return classification

# clf = MeanShift(radius=3)    Knowing correct radius is hard for completely unknown data
# plt.scatter(X[:, 0], X[:, 1], s=50)
clf = DynamicMeanShift()
clf.fit(X)
centroids = clf.centroids

for classification in clf.classifications:
    color = colours[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=15, linewidths=5)

for c in centroids:
    color = colours[c]
    plt.scatter(centroids[c][0], centroids[c][1], color=color, marker='*', s=150, linewidths=2)

plt.show()
