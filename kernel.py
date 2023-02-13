import numpy as np
import math


def linear(x, y):
    return np.dot(x, y)


def polynomial(x, y, p):
    return ((np.dot(x, y) + 1) ** p)


def radial_basis_function(x, y, sigma):
    return math.exp(-np.abs(x-y)**2 / (2 * sigma**2))
