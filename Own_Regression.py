"""
This file contains code in writing our own code to do linear regression on data and calculating r^2 values.
"""

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import random
from statistics import mean

style.use('ggplot')


# Before we start writing linear regression and coefficient of determination code, we write code to automatically
# generate data to test our code
def create_dataset(datapoints, variance, step=2, correlation=False):
    """
    Creates a dataset of x and y values to test out best fit code

    :param datapoints: Number of datapoints to have in our graph
    :param variance: How much each point can vary from previous point
    :param step: How far to step for each point (vertically along y axis, this creates correlation)
    :param correlation:  False/pos/neg to give no/positive/negative correlation to generated dataset
    :return: (x values, y values) as numpy arrays
    """
    val = 1     # Starting value
    xdata = [i for i in range(datapoints)]  # x values go up by one steps
    ydata = []
    for i in range(datapoints):
        y = val + random.randrange(-variance, variance)     # Generate a y value with data val +- variance
        ydata.append(y)
        # Next step adds correlation by affecting where next y value will be placed
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    return np.array(xdata, dtype=np.float64), np.array(ydata, dtype=np.float64)


'''First we work out the line of best fit for our data'''

xs, ys = create_dataset(40, 5, 2, correlation=False)
# xs = [1, 2, 3, 4, 5]    # For manual dataset
# ys = [5, 4, 6, 5, 6]
# plt.scatter(xs, ys)     # To visualise the data
# plt.show()

# Reason for using numpy array is so that we can make use of e.g. matrix multiplications built in numpy
# and also numpy representation is very memory efficient, it sacrifices Python's List extensibility (any object can be
# in a list) for efficient storage.
xs = np.array(xs, dtype=np.float64)  # We change Python lists to numpy array with explicit data type (for future)
ys = np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xdata, ydata):
    """Returns (slope, y-intercept) for line of best fit"""
    mm = (((mean(xdata) * mean(ydata)) - mean(xdata * ydata)) /
         ((mean(xdata) ** 2) - mean(xdata ** 2)))
    cc = mean(ydata) - (mm * mean(xdata))
    return mm, cc

m, c = best_fit_slope_and_intercept(xs, ys)
# Given m and c, we can now predict what y value will be for any given x
predict_x = 8
predict_y = (m * predict_x) + c
regression_line = [(m * x) + c for x in xs]                # Generate y axis data for best fit from x, m and c values

plt.scatter(xs, ys, label='data')                          # Plot the existing x and y values
plt.plot(xs, regression_line, label='regression line')     # Plot the regression line
plt.scatter(predict_x, predict_y, color='g')               # Plot the predicted point in graph as green point
plt.legend(loc=4)                                          # Show the added labels as Legend in 4th Quadrant
plt.show()

'''Now we see how accurate our line of best fit is to the data, using r-squared'''


def squared_error(ys_orig, ys_line):
    """Returns the squared error for entire line. We pass in array of y original values and array of values from line"""
    return sum((ys_line - ys_orig) ** 2)  # Syntax works since y values are numpy arrays!. Doesn't work for Python lists


def coefficient_of_determination(ys_orig, ys_line):
    """Returns the r^2 value for line of best fit. We pass in array of y values for both parameters"""
    y_mean_line = [mean(ys_orig) for _ in ys_orig]              # We have same y value for all x-value, i.e y mean value
    squared_error_regression = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regression / squared_error_y_mean)

r_squared = coefficient_of_determination(ys, regression_line)
print("Coefficient of determination: ", r_squared)
