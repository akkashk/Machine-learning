import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class SupportVectorMachine:
    """This is made a class to save the trained state of classifier"""
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.colours = {1: 'r', -1: 'b'}
        if self.visualisation:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        """Trains the classifier on the data. Finds self.w and self.b values for prediction"""
        self.data = data
        opt_dict = {}   # Contains magnitude of w as key with w and b values {||w||: (w, b)}
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]   # Applied to w vector to try all directions

        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None     # To clear memory

        # Step sizes to take when finding global minimum, start big and change to smaller steps.
        # Smaller the step, more performance cost
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      self.max_feature_value * 0.001]
        b_range_multiple = 5    # Value of self.b does not need to be as accurate, so can take larger, fixed steps
        b_multiple = 5

        # First guess at self.w value.
        latest_optimum = self.max_feature_value * 10

        '''NOTE: In this algo all dimensions of w will have the same value when optimising.
        This is not the case in real life and only done here to get w value faster'''

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])  # In this algo, we only use 2D vector space
            optimised = False   # We can know this since convex problem, otherwise we don't when when at global minimum
            while not optimised:
                # Need to optimise w and b value
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True  # Treat each w as potential optimum answer
                        for yi in self.data:  # Need to run this config across ALL data to check. Problem of SVM!
                            for xi in self.data[yi]:  # yi is the label in data
                                if not yi*(np.dot(xi, w_t) + b) >= 1:  # Condition: yi(xi.w + b) >= 1
                                    found_option = False    # If codition does not hold, this config cannot be for w & b

                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = (w_t, b)

                if w[0] < 0:
                    optimised = True
                    print("Optimised a step.")
                else:
                    w = w - step    # Step down on w vector by 'step' amount
            norms = sorted([n for n in opt_dict])   # Norms is sorted list of magnitudes in ascending order
            opt_choice = opt_dict[norms[0]]         # We choose the least magnitude value
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            # Change latest optimum to reflect on new found optimum. step*2 so we start at a higher guess and then step
            latest_optimum = opt_choice[0][0] + step*2

            # To print data to see where data lies
            for i in self.data:
                for xi in self.data[i]:
                    yi = i
                    print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

    def predict(self, features):
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)   # sign of x.w + b, x is feature vector
        if classification != 0 and self.visualisation:
            self.ax.scatter(features[0], features[1], s=200, marker='*', color=self.colours[classification])
        return classification

    def visualise(self):
        """
        Just to see where the support vectors and decision boundaries are when classifying, actual algo doesn't use
        these.
        """
        [[self.ax.scatter(x[0], x[1], s=50, color=self.colours[i]) for x in data_dict[i]] for i in data_dict]

        def hyperplane(x, w, b, v):
            """Definition of hyperplane is x.w + b = v.
            In Positive support vector, v = 1. In Negative support vector, v = -1. In decision boundary, v = 0."""
            return (-w[0] * x - b + v) / w[1]

        datarange = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyperplane_x_min = datarange[0]
        hyperplane_x_max = datarange[1]

        # Positive support vector hyperplane, (w.x + b) = 1
        psv1 = hyperplane(hyperplane_x_min, self.w, self.b, 1)  # psv1 is just a single, scalar value
        psv2 = hyperplane(hyperplane_x_max, self.w, self.b, 1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [psv1, psv2], 'k')

        # Negative support vector hyperplane, (w.x + b) = -1
        nsv1 = hyperplane(hyperplane_x_min, self.w, self.b, -1)  # psv1 is just a single, scalar value
        nsv2 = hyperplane(hyperplane_x_max, self.w, self.b, -1)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [nsv1, nsv2], 'k')

        # Decision boundary vector hyperplane, (w.x + b) = 0
        db1 = hyperplane(hyperplane_x_min, self.w, self.b, 0)  # psv1 is just a single, scalar value
        db2 = hyperplane(hyperplane_x_max, self.w, self.b, 0)
        self.ax.plot([hyperplane_x_min, hyperplane_x_max], [db1, db2], 'y--')

        plt.show()


# Each key corresponds to label attached and each sub-list represents value of features for that label
data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])
             }
svm = SupportVectorMachine()
svm.fit(data_dict)
predict_us = [[0, 10],
              [1, 3],
              [3, 4],
              [3, 5],
              [5, 5],
              [5, 6],
              [6, -5],
              [5, 8]]
for p in predict_us:
    svm.predict(p)  # These points will be shown as colour-coded '*'s
svm.visualise()

'''
POTENTIAL OPTIMISATIONS:
i)   Have a threshold above with support vector accuracies need to be to draw OR get the one closest to it
ii)  Running with only two step sizes, we can see support vector hyperplanes don't go thru actual data points, draw
     boundary hyperplane wrong and wrongly predicts test points, have at least the 3 we have
iii) Using different b_range values. This is a MAJOR optimisation in terms of time taken.
     b_range_multiple most expensive, b_multiple less so
iv)  Break for loop when we know config can't be used
v)   The 'while not optimised' loop CAN be threaded to improve performance. We can't thread the step size changes as we
     need info from previous step size on where to make finer steps
'''