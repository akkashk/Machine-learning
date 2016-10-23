import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.show()

colours = ['g', 'r', 'c', 'b', 'k']


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        """
        :param k:  Number of clusters to find
        :param tol: How much centroid needs to move below between iterations to stop
        :param max_iter: Mai number of iterations after which we stop
        """
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = {}
        self.classifications = {}

    def fit(self, data):
        for i in range(self.k):
            self.centroids[i] = data[i]     # Centroids will be first i feature sets from data, can be set randomly too

        for i in range(self.max_iter):      # We start optimisation
            self.classifications = {}       # For every iteration, we reclassify, so need to make it empty

            for j in range(self.k):
                self.classifications[j] = []    # Key = centroids, Values = Featuresets closest to centroid

            for featureset in data:
                # Find distance between feature set and all centroid, length of distances = number of centroids
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                # Classify feature set to closest centroid, index() returns index in which min(distance) saved
                classification = distances.index(min(distances))      # Closest centroid saved in classification
                # Append feature set to the list of feature sets closets to centroid
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)   # To create a copy of values, and not refer to same dictionary

            for classification in self.classifications:
                # Redefine centroid by taking mean
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimised = True
            for centroid in self.centroids:
                original_centroid = prev_centroids[centroid]
                current_centroid = self.centroids[centroid]
                if np.sum(((current_centroid - original_centroid) / original_centroid) * 100.0) > self.tol:
                    optimised = False
            if optimised:
                print("Number of iterations done: ", i)
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = KMeans()
clf.fit(X)

for c in clf.centroids:
    plt.scatter(clf.centroids[c][0], clf.centroids[c][1], marker='o', color='k', linewidths=5)

for c in clf.classifications:
    colour = colours[c]
    for featureset in clf.classifications[c]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=colour, linewidths=5)

unknowns = np.array([[1, 3],
                    [4, 4],
                    [10, 10],
                    [5, 5]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colours[classification], s=50, linewidths=5)

plt.show()
