from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

def create_dataset(length, variance, step=2, correlation='pos'):
    val = 1
    ys = []
    for i in range(length):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(length)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs*xs))
    return m

def best_fit_slope_2(xs, ys):
    numenator = sum((ys - mean(ys)) * (xs - mean(xs)))
    denominator = sum((xs - mean(xs))**2)
    m = numenator / denominator
    return m

def best_fit_intercept(xs, ys, m):
    b = mean(ys) - (m * mean(xs))
    return b

def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line)**2)

def coefficient_of_determination(ys_orig, ys_regr):
    y_mean_line = [mean(ys_orig)] * len(ys_orig)
    squared_error_regr = squared_error(ys_orig, ys_regr)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    print(squared_error_regr, "/", squared_error_y_mean)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 20, 2, correlation='pos')

m = best_fit_slope_2(xs, ys)
b = best_fit_intercept(xs, ys, m)
regression_line = [(m * x) + b for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()