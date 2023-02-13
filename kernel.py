import numpy as np
import math


def linear(x, y):
    return np.dot(x, y)

def polynomial(p):
    def poly(x, y):
        return ((np.dot(x, y) + 1) ** p)
    return poly


def radial_basis_function(sigma):
    def radial(x, y):
        return math.exp(-np.linalg.norm(x-y)**2 / (2 * sigma**2))
    return radial
